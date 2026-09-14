// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_core::{DigestSha256, StableId};
use symthaea_assurance_semantics::{
    SemanticCommitmentError, SemanticCommitmentV1, canonical_semantic_set,
};

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
) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
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
fn exact_components_are_preserved() {
    let value = commitment("matched-sham", "canonical-text-v1", 'a');
    assert_eq!(value.semantic_id().as_str(), "matched-sham");
    assert_eq!(value.definition_schema().as_str(), "canonical-text-v1");
    assert_eq!(value.definition_digest().as_str(), "a".repeat(64));
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
