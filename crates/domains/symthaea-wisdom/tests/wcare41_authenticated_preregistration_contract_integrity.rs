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
fn wcare41_contract_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        ("docs/release/evidence/WCARE41_AUTHENTICATED_PREREGISTRATION_PROTOCOL_V1.md", "8df5bbe387f4c221d55aa2ac16203b57ed602779"),
        ("docs/release/evidence/WCARE41_AUTHENTICATION_PLAN_SCHEMA_V1.json", "32683abe68e44a1b1bfce2b9c08e2bcd7041877f"),
        ("docs/release/evidence/WCARE41_BUILDER_ATTESTATION_ENVELOPE_SCHEMA_V1.json", "e2ffc9b452378120ed0cb40ae459bab8b9a76cde"),
        ("docs/release/evidence/WCARE41_TEMPORAL_PROOF_PACKAGE_SCHEMA_V1.json", "2771b0af9b319a697716ff5d195724f42c222897"),
        ("docs/release/evidence/WCARE41_BUILDER_ISSUER_TRUST_POLICY_SCHEMA_V1.json", "27ca0805dc041b3573b10f4accdee8db40a4e6de"),
        ("docs/release/evidence/WCARE41_TEMPORAL_VERIFIER_POLICY_SCHEMA_V1.json", "4d2acc1ac5fb4332bdcb934e35284d21f6957e41"),
        ("docs/release/evidence/WCARE41_AUTHENTICATION_RESULT_SCHEMA_V1.json", "17cecb49abea363e7c93a4ce8b3f3ed22910eb1f"),
        ("scripts/wcare41_contract_selftest.py", "d428422aa5a5c65e5c108a513034f67a4d8a9bfa"),
        ("scripts/wcare41-integrity.sh", "fab0c3188564b6f343acff796e8f45622db0c962"),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-41 byte drift: {path}");
    }
}

#[test]
fn wcare41_contract_campaign_executes_without_authentication_claims() {
    let root = repo_root();
    let output = Command::new("python3")
        .arg("scripts/wcare41_contract_selftest.py")
        .current_dir(&root)
        .output()
        .expect("run WCARE-41 contract campaign");
    assert!(
        output.status.success(),
        "WCARE-41 contract campaign failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("self-test stdout is utf-8");
    for marker in [
        "PASS_WCARE41_CONTRACT_SELFTEST",
        "\"cryptographic_authentication_executed\":false",
        "\"external_temporal_verification_executed\":false",
        "\"builder_temporal_orthogonality_verified\":true",
        "\"late_commitment_rejected_as_preregistration\":true",
        "\"duplicate_evidence_does_not_multiply_coverage\":true",
        "\"authentication_monotonicity_verified\":true",
    ] {
        assert!(stdout.contains(marker), "missing WCARE-41 marker {marker}: {stdout}");
    }
}
