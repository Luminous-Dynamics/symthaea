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
fn wcare42_prelock_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "tools/wcare42_builder_attestation_verifier/Cargo.toml",
            "5410040e5616241dd4ba581af8f297675d083830",
        ),
        (
            "tools/wcare42_builder_attestation_verifier/src/main.rs",
            "1c300a455f054d118e55556aac81b623824629bc",
        ),
        (
            "tools/wcare42_builder_attestation_verifier/tests/golden.rs",
            "666242f74fb302f9be2b62fa4b3050f3ee9ffefd",
        ),
        (
            "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_VERIFIER_PROTOCOL_V1.md",
            "1cd6e3f5f2f4edcf4fccf45b35c4c68bdb9c12a5",
        ),
        (
            "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_RESULT_SCHEMA_V1.json",
            "b2d6b4b61af46c8924cdb95b2f958df3d3d7ab96",
        ),
        (
            "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_GOLDEN_VECTOR_V1.json",
            "dc42f404537795f37a9bf178d3f30fd52c74a355",
        ),
        (
            "scripts/wcare42-qualify.sh",
            "8a80d1a74e409503a72b4607425cc61d8072ca2e",
        ),
        (
            "scripts/wcare42-integrity.sh",
            "ecb602b818619c4e091b4878943d5b760931f327",
        ),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-42 byte drift: {path}");
    }
}

#[test]
fn wcare42_prelock_campaign_proves_fail_closed_blocker() {
    let root = repo_root();
    let output = Command::new("bash")
        .arg("scripts/wcare42-integrity.sh")
        .current_dir(&root)
        .output()
        .expect("run WCARE-42 pre-lock integrity campaign");
    assert!(
        output.status.success(),
        "WCARE-42 pre-lock campaign failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("integrity stdout is utf-8");
    for marker in [
        "PASS_SOURCE_INTEGRITY",
        "\"cryptographic_execution_qualified\":false",
        "\"builder_authentication_established\":false",
        "\"preregistration_temporal_precedence_established\":false",
        "\"runtime_authority_granted\":false",
    ] {
        assert!(stdout.contains(marker), "missing WCARE-42 marker {marker}: {stdout}");
    }
}
