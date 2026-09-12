// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use sha2::{Digest, Sha256};
use symthaea_replicator_semantics::{
    BoundCapabilitySet, CapabilitySchemaId, ResourceAccountingSchemeId, ResourceDimensionId,
    ResourceQuantity, ResourceVector, SemanticBindingError,
};

const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_1.json"
));

fn digest_value(value: &Value) -> String {
    let canonical = serde_json::to_vec(value).expect("golden value must serialize");
    hex::encode(Sha256::digest(canonical))
}

fn digest_bytes(hex_digest: &str) -> [u8; 32] {
    hex::decode(hex_digest)
        .expect("golden digest must be hex")
        .try_into()
        .expect("golden digest must be exactly 32 bytes")
}

fn quantity(id: u16, amount: u64) -> ResourceQuantity {
    ResourceQuantity::new(ResourceDimensionId::new(id), amount)
}

#[test]
fn rust_recomputes_cross_language_golden_schema_ids() {
    let root: Value = serde_json::from_str(GOLDEN).expect("golden corpus must parse");
    assert_eq!(
        digest_value(&root["capability_schema"]),
        root["capability_schema_sha256"].as_str().unwrap()
    );
    assert_eq!(
        digest_value(&root["resource_schema"]),
        root["resource_schema_sha256"].as_str().unwrap()
    );
}

#[test]
fn same_bits_under_different_capability_schema_are_incomparable() {
    let left_schema = CapabilitySchemaId::new([1; 32]);
    let right_schema = CapabilitySchemaId::new([2; 32]);
    let left = BoundCapabilitySet::new(left_schema, 0b11);
    let right = BoundCapabilitySet::new(right_schema, 0b11);

    assert_eq!(
        left.is_subset_of(right),
        Err(SemanticBindingError::CapabilitySchemaMismatch {
            left: left_schema,
            right: right_schema,
        })
    );
    assert_eq!(
        left.intersect(right),
        Err(SemanticBindingError::CapabilitySchemaMismatch {
            left: left_schema,
            right: right_schema,
        })
    );
}

#[test]
fn same_schema_capability_operations_preserve_schema() {
    let schema = CapabilitySchemaId::new([3; 32]);
    let broad = BoundCapabilitySet::new(schema, 0b111);
    let narrow = BoundCapabilitySet::new(schema, 0b011);

    assert!(narrow.is_subset_of(broad).unwrap());
    assert_eq!(
        broad.intersect(narrow).unwrap(),
        BoundCapabilitySet::new(schema, 0b011)
    );
}

#[test]
fn resource_vectors_require_canonical_unique_dimensions() {
    let scheme = ResourceAccountingSchemeId::new([4; 32]);
    assert_eq!(
        ResourceVector::new(scheme, vec![]),
        Err(SemanticBindingError::EmptyResourceVector)
    );
    assert_eq!(
        ResourceVector::new(scheme, vec![quantity(1, 1), quantity(1, 2)]),
        Err(SemanticBindingError::NonCanonicalResourceDimensions)
    );
    assert_eq!(
        ResourceVector::new(scheme, vec![quantity(2, 1), quantity(1, 2)]),
        Err(SemanticBindingError::NonCanonicalResourceDimensions)
    );
}

#[test]
fn resource_arithmetic_rejects_scheme_or_dimension_substitution() {
    let a = ResourceAccountingSchemeId::new([5; 32]);
    let b = ResourceAccountingSchemeId::new([6; 32]);
    let left = ResourceVector::new(a, vec![quantity(0, 10), quantity(1, 20)]).unwrap();
    let wrong_scheme = ResourceVector::new(b, vec![quantity(0, 1), quantity(1, 2)]).unwrap();
    let wrong_dimensions = ResourceVector::new(a, vec![quantity(0, 1), quantity(2, 2)]).unwrap();

    assert_eq!(
        left.checked_add(&wrong_scheme),
        Err(SemanticBindingError::ResourceSchemeMismatch { left: a, right: b })
    );
    assert_eq!(
        left.checked_add(&wrong_dimensions),
        Err(SemanticBindingError::ResourceDimensionSetMismatch)
    );
}

#[test]
fn golden_resource_remaining_matches_reference_corpus() {
    let root: Value = serde_json::from_str(GOLDEN).expect("golden corpus must parse");
    let scheme = ResourceAccountingSchemeId::new(digest_bytes(
        root["resource_schema_sha256"].as_str().unwrap(),
    ));
    let limits = &root["resource_vectors"][0]["amounts"];
    let consumed = &root["resource_vectors"][1]["amounts"];
    let expected = &root["expected_remaining"];

    let limit = ResourceVector::new(
        scheme,
        vec![
            quantity(0, limits["budget.compute"].as_u64().unwrap()),
            quantity(1, limits["budget.energy"].as_u64().unwrap()),
        ],
    )
    .unwrap();
    let used = ResourceVector::new(
        scheme,
        vec![
            quantity(0, consumed["budget.compute"].as_u64().unwrap()),
            quantity(1, consumed["budget.energy"].as_u64().unwrap()),
        ],
    )
    .unwrap();
    let remaining = limit.checked_remaining(&used).unwrap();

    assert_eq!(
        remaining.quantities(),
        &[
            quantity(0, expected["budget.compute"].as_u64().unwrap()),
            quantity(1, expected["budget.energy"].as_u64().unwrap()),
        ]
    );
}

#[test]
fn resource_underflow_overflow_and_transition_widening_fail_closed() {
    let scheme = ResourceAccountingSchemeId::new([7; 32]);
    let low = ResourceVector::new(scheme, vec![quantity(0, 1)]).unwrap();
    let high = ResourceVector::new(scheme, vec![quantity(0, 2)]).unwrap();
    assert_eq!(
        low.checked_remaining(&high),
        Err(SemanticBindingError::ResourceUnderflow {
            dimension: ResourceDimensionId::new(0),
        })
    );

    let max = ResourceVector::new(scheme, vec![quantity(0, u64::MAX)]).unwrap();
    assert_eq!(
        max.checked_add(&low),
        Err(SemanticBindingError::ResourceOverflow {
            dimension: ResourceDimensionId::new(0),
        })
    );

    let envelope = ResourceVector::new(scheme, vec![quantity(0, 5)]).unwrap();
    let narrower = ResourceVector::new(scheme, vec![quantity(0, 4)]).unwrap();
    let wider = ResourceVector::new(scheme, vec![quantity(0, 6)]).unwrap();
    ResourceVector::verify_conservative_remaining_transition(&narrower, &envelope).unwrap();
    assert_eq!(
        ResourceVector::verify_conservative_remaining_transition(&wider, &envelope),
        Err(SemanticBindingError::TransitionExceedsConservativeEnvelope {
            dimension: ResourceDimensionId::new(0),
        })
    );
}
