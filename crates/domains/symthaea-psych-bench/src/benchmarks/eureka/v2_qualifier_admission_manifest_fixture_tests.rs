// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent replay of the checked-in admission-manifest signing fixture.

use super::v2_qualifier_admission_manifest::{
    V2QualifierAdmissionKind, V2QualifierAdmissionManifest, V2QualifierAdmissionRevisions,
};

const GENESIS_GOLDEN: &[u8] = include_bytes!(
    "fixtures/eureka-v2-qualifier-admission-manifest-v1-genesis.env"
);

fn hex(byte: char) -> String {
    byte.to_string().repeat(64)
}

fn head(byte: char) -> String {
    byte.to_string().repeat(40)
}

#[test]
fn independent_fixture_matches_exact_canonical_signing_bytes() {
    let manifest = V2QualifierAdmissionManifest::from_hex(
        V2QualifierAdmissionKind::Genesis,
        &head('1'),
        &head('2'),
        V2QualifierAdmissionRevisions::current(
            "EUREKA.002.V2.QUALIFIER_ENVIRONMENT.v1",
        )
        .unwrap(),
        1,
        None,
        None,
        &hex('a'),
        1,
        &hex('b'),
        &hex('c'),
        &hex('d'),
        &hex('e'),
        &hex('f'),
    )
    .unwrap();

    assert_eq!(manifest.canonical_bytes(), GENESIS_GOLDEN);
    assert_ne!(manifest.commitment(), [0_u8; 32]);
}

#[test]
fn golden_fixture_contains_no_provenance_only_or_execution_fields() {
    let text = std::str::from_utf8(GENESIS_GOLDEN).unwrap();
    for forbidden in [
        "timestamp=",
        "reviewer=",
        "github_run_id=",
        "github_run_attempt=",
        "pull_request=",
        "artifact_url=",
        "execution_authority_granted=true",
    ] {
        assert!(!text.contains(forbidden), "forbidden golden field: {forbidden}");
    }
    assert!(text.ends_with("execution_authority_granted=false\n"));
}
