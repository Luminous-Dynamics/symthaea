// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/semantic_schema.rs"]
mod semantic_schema;

use semantic_schema::{
    OrderingLineageIdentityV1, SelfDescribingSemanticCommitmentV1, SemanticCommitmentError,
    canonical_semantic_set,
};
use symthaea_assurance_core::{DigestSha256, StableId};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn commitment(
    semantic_id: &str,
    definition_schema: &str,
    definition_byte: char,
) -> SelfDescribingSemanticCommitmentV1 {
    SelfDescribingSemanticCommitmentV1::new(
        id(semantic_id),
        id(definition_schema),
        digest(definition_byte),
    )
}

#[test]
fn definition_schema_is_identity_bearing() {
    let left = commitment("matched-sham", "canonical-text-v1", 'a');
    let right = commitment("matched-sham", "canonical-json-v1", 'a');
    assert_ne!(left, right);
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn definition_digest_is_identity_bearing() {
    let left = commitment("matched-sham", "canonical-text-v1", 'a');
    let right = commitment("matched-sham", "canonical-text-v1", 'b');
    assert_ne!(left, right);
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn duplicate_semantic_id_fails_even_when_schema_differs() {
    let error = canonical_semantic_set(vec![
        commitment("same-control", "canonical-text-v1", 'a'),
        commitment("same-control", "canonical-json-v1", 'a'),
    ])
    .unwrap_err();
    assert_eq!(
        error,
        SemanticCommitmentError::DuplicateSemanticId("same-control".into())
    );
}

#[test]
fn canonical_semantic_set_orders_by_semantic_id_only() {
    let ordered = canonical_semantic_set(vec![
        commitment("zeta", "schema-z", 'c'),
        commitment("alpha", "schema-a", 'a'),
        commitment("middle", "schema-m", 'b'),
    ])
    .unwrap();
    let ids: Vec<_> = ordered
        .iter()
        .map(|value| value.semantic_id().as_str())
        .collect();
    assert_eq!(ids, vec!["alpha", "middle", "zeta"]);
}

#[test]
fn validation_profile_schema_change_makes_ordering_lineage_incomparable() {
    let left = OrderingLineageIdentityV1::new(
        id("transparency-log-a"),
        commitment("ordering-profile", "canonical-text-v1", 'a'),
        1,
    );
    let right = OrderingLineageIdentityV1::new(
        id("transparency-log-a"),
        commitment("ordering-profile", "canonical-json-v1", 'a'),
        1,
    );
    assert!(!left.is_comparable_with(&right));
}

#[test]
fn ordering_lineage_requires_source_profile_and_epoch_identity() {
    let profile = commitment("ordering-profile", "canonical-text-v1", 'a');
    let baseline = OrderingLineageIdentityV1::new(id("transparency-log-a"), profile.clone(), 1);
    let same = OrderingLineageIdentityV1::new(id("transparency-log-a"), profile.clone(), 1);
    let other_source = OrderingLineageIdentityV1::new(id("transparency-log-b"), profile.clone(), 1);
    let other_epoch = OrderingLineageIdentityV1::new(id("transparency-log-a"), profile, 2);
    assert!(baseline.is_comparable_with(&same));
    assert!(!baseline.is_comparable_with(&other_source));
    assert!(!baseline.is_comparable_with(&other_epoch));
}

#[test]
fn independent_canonical_golden_vector_is_stable() {
    let commitment = commitment(
        "matched-sham",
        "symthaea.assurance.semantic-definition.canonical-text-v1",
        'a',
    );
    assert_eq!(
        commitment.digest().as_str(),
        "ccf57b5d5fbc762fc072eaa7b885a2eff8e8de78a56329e4999b403da2011ac4"
    );
}
