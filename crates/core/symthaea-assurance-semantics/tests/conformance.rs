// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_core::{DigestSha256, StableId};
use symthaea_assurance_semantics::{
    DefinitionSchemaV1, SemanticCommitmentError, SemanticCommitmentV1, canonical_semantic_set,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn schema(schema_id: &str, specification_byte: char) -> DefinitionSchemaV1 {
    DefinitionSchemaV1::new(id(schema_id), digest(specification_byte))
}

fn commitment(
    semantic_id: &str,
    definition_schema: &str,
    schema_specification_byte: char,
    definition_byte: char,
) -> SemanticCommitmentV1 {
    SemanticCommitmentV1::new(
        id(semantic_id),
        schema(definition_schema, schema_specification_byte),
        digest(definition_byte),
    )
}

#[test]
fn definition_schema_id_is_identity_bearing() {
    let left = commitment("matched-sham", "canonical-text-v1", '1', 'a');
    let right = commitment("matched-sham", "canonical-json-v1", '1', 'a');
    assert_ne!(left, right);
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn definition_schema_specification_is_identity_bearing() {
    let left = commitment("matched-sham", "canonical-text-v1", '1', 'a');
    let right = commitment("matched-sham", "canonical-text-v1", '2', 'a');
    assert_ne!(left, right);
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn definition_digest_is_identity_bearing() {
    let left = commitment("matched-sham", "canonical-text-v1", '1', 'a');
    let right = commitment("matched-sham", "canonical-text-v1", '1', 'b');
    assert_ne!(left, right);
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn exact_components_are_preserved() {
    let value = commitment("matched-sham", "canonical-text-v1", '1', 'a');
    assert_eq!(value.semantic_id().as_str(), "matched-sham");
    assert_eq!(
        value.definition_schema().schema_id().as_str(),
        "canonical-text-v1"
    );
    assert_eq!(
        value.definition_schema().specification_digest().as_str(),
        "1".repeat(64)
    );
    assert_eq!(value.definition_digest().as_str(), "a".repeat(64));
}

#[test]
fn duplicate_semantic_id_fails_even_when_schema_differs() {
    let error = canonical_semantic_set(vec![
        commitment("same-control", "canonical-text-v1", '1', 'a'),
        commitment("same-control", "canonical-json-v1", '2', 'a'),
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
        commitment("zeta", "schema-z", '3', 'c'),
        commitment("alpha", "schema-a", '1', 'a'),
        commitment("middle", "schema-m", '2', 'b'),
    ])
    .unwrap();
    let ids: Vec<_> = ordered
        .iter()
        .map(|value| value.semantic_id().as_str())
        .collect();
    assert_eq!(ids, vec!["alpha", "middle", "zeta"]);
}

#[test]
fn canonical_lengths_are_utf8_byte_lengths() {
    let value = commitment("café", "schema-β", '1', 'a');
    let canonical = String::from_utf8(value.canonical_bytes()).unwrap();
    assert!(canonical.contains("semantic-id 5:café\n"));
    assert!(canonical.contains("definition-schema-id 9:schema-β\n"));
}

#[test]
fn independent_canonical_golden_vector_is_stable() {
    let commitment = commitment(
        "matched-sham",
        "symthaea.assurance.semantic-definition.canonical-text-v1",
        'b',
        'a',
    );
    assert_eq!(
        commitment.digest().as_str(),
        "3a510b215853c77285e39c95b6ab45c0d816370e17e14ccda8337110a790f1ba"
    );
}
