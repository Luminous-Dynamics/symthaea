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
fn wcare43_exact_review_unit_is_frozen() {
    let root = repo_root();
    for (path, expected) in [
        (
            "docs/release/evidence/WCARE43_RFC3161_PREREGISTRATION_PROTOCOL_V1.md",
            "76ff1774ee510c096649b82cfc3857deaf921e12",
        ),
        (
            "docs/release/evidence/WCARE43_RFC3161_BACKEND_POLICY_SCHEMA_V1.json",
            "2e04e221a5d89bd2da3ea4244649898b66e47a74",
        ),
        (
            "docs/release/evidence/WCARE43_RFC3161_RESULT_SCHEMA_V1.json",
            "7d67eec8adc535d37b4a62c748ad24d3b02d30d0",
        ),
        (
            "scripts/wcare43_rfc3161_verify.py",
            "e89b66a24cb51bed589911b082ea6fef5ffd369b",
        ),
        (
            "scripts/wcare43_selftest.py",
            "7906ce87f7b121464b42dbf35584ddbc9a50ae4d",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/a.final.json",
            "8cec0486532180e6f4fe0f58920c1f318b5154ac",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/b.final.json",
            "ba7cce8fbfece6812fa44816942ed66d3ed7e55f",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/fixture_manifest.json",
            "ec601cc591365dd6f9026894c477ae5a7180c97a",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/synthetic_plan.json",
            "a6569381df8e320c286833a9830075376482206a",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/synthetic_response.tsr.b64",
            "0add55553bbe1927635b747b184523b0e87e576f",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/synthetic_root.pem",
            "901a385c49aa56a5606324319a38956cba414fc1",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/synthetic_tsa.pem",
            "d4ca7537a5baebe978e8cc3744763716a5bb800e",
        ),
        (
            "docs/release/evidence/fixtures/wcare43/synthetic_wcare40_result.json",
            "8a48ce8597167d4de32b67806217e6a0525267cd",
        ),
        (
            "scripts/wcare43-integrity.sh",
            "93e52ac22fbe262d3f3ead5a94fa95b7ffcc238a",
        ),
    ] {
        assert_eq!(git_blob(&root, path), expected, "WCARE-43 byte drift: {path}");
    }
}

#[test]
fn wcare43_rfc3161_campaign_executes_without_external_claim() {
    let root = repo_root();
    let output = Command::new("bash")
        .arg("scripts/wcare43-integrity.sh")
        .current_dir(&root)
        .output()
        .expect("run WCARE-43 integrity gate");
    assert!(
        output.status.success(),
        "WCARE-43 integrity gate failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("integrity stdout is utf-8");
    assert!(stdout.contains("PASS_PROTOCOL_INTEGRITY"), "missing WCARE-43 PASS marker: {stdout}");
    assert!(
        stdout.contains("\"real_external_preregistration_established\":false"),
        "synthetic fixture overclaimed external preregistration: {stdout}"
    );
}
