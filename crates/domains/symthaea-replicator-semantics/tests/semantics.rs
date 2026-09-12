// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_replicator_semantics::{
    BoundCapabilitySet, CapabilitySchemaId, ResourceAccountingSchemeId, ResourceDimensionId,
    ResourceQuantity, ResourceVector, SemanticBindingError,
};

fn quantity(id: u16, amount: u64) -> ResourceQuantity {
    ResourceQuantity::new(ResourceDimensionId::new(id), amount)
}

#[test]
fn capability_operations_require_exact_schema_identity() {
    let a = CapabilitySchemaId::new([1; 32]);
    let b = CapabilitySchemaId::new([2; 32]);
    let left = BoundCapabilitySet::new(a, 0b11);
    let same = BoundCapabilitySet::new(a, 0b01);
    let other = BoundCapabilitySet::new(b, 0b01);

    assert!(same.is_subset_of(left).unwrap());
    assert_eq!(
        left.intersect(same).unwrap(),
        BoundCapabilitySet::new(a, 0b01)
    );
    assert_eq!(
        left.is_subset_of(other),
        Err(SemanticBindingError::CapabilitySchemaMismatch { left: a, right: b })
    );
}

#[test]
fn resource_vectors_require_strict_canonical_dimension_order() {
    let scheme = ResourceAccountingSchemeId::new([3; 32]);
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
fn checked_remaining_is_per_dimension_and_fail_closed() {
    let scheme = ResourceAccountingSchemeId::new([4; 32]);
    let limit = ResourceVector::new(scheme, vec![quantity(0, 100), quantity(1, 80)]).unwrap();
    let consumed = ResourceVector::new(scheme, vec![quantity(0, 40), quantity(1, 20)]).unwrap();
    let remaining = limit.checked_remaining(&consumed).unwrap();
    assert_eq!(remaining.quantities(), &[quantity(0, 60), quantity(1, 60)]);

    let over = ResourceVector::new(scheme, vec![quantity(0, 101), quantity(1, 20)]).unwrap();
    assert_eq!(
        limit.checked_remaining(&over),
        Err(SemanticBindingError::ResourceUnderflow {
            dimension: ResourceDimensionId::new(0),
        })
    );
}

#[test]
fn resource_arithmetic_rejects_scheme_and_dimension_substitution() {
    let a = ResourceAccountingSchemeId::new([5; 32]);
    let b = ResourceAccountingSchemeId::new([6; 32]);
    let base = ResourceVector::new(a, vec![quantity(0, 10), quantity(1, 20)]).unwrap();
    let wrong_scheme = ResourceVector::new(b, vec![quantity(0, 1), quantity(1, 2)]).unwrap();
    let wrong_dimensions = ResourceVector::new(a, vec![quantity(0, 1), quantity(2, 2)]).unwrap();

    assert_eq!(
        base.checked_add(&wrong_scheme),
        Err(SemanticBindingError::ResourceSchemeMismatch { left: a, right: b })
    );
    assert_eq!(
        base.checked_add(&wrong_dimensions),
        Err(SemanticBindingError::ResourceDimensionSetMismatch)
    );
}

#[test]
fn conservative_envelope_and_overflow_cannot_widen_authority() {
    let scheme = ResourceAccountingSchemeId::new([7; 32]);
    let envelope = ResourceVector::new(scheme, vec![quantity(0, 5), quantity(1, 7)]).unwrap();
    let narrower = ResourceVector::new(scheme, vec![quantity(0, 4), quantity(1, 7)]).unwrap();
    let wider = ResourceVector::new(scheme, vec![quantity(0, 6), quantity(1, 7)]).unwrap();

    ResourceVector::verify_conservative_remaining_transition(&narrower, &envelope).unwrap();
    assert_eq!(
        ResourceVector::verify_conservative_remaining_transition(&wider, &envelope),
        Err(SemanticBindingError::TransitionExceedsConservativeEnvelope {
            dimension: ResourceDimensionId::new(0),
        })
    );

    let max = ResourceVector::new(scheme, vec![quantity(0, u64::MAX), quantity(1, 0)]).unwrap();
    let one = ResourceVector::new(scheme, vec![quantity(0, 1), quantity(1, 0)]).unwrap();
    assert_eq!(
        max.checked_add(&one),
        Err(SemanticBindingError::ResourceOverflow {
            dimension: ResourceDimensionId::new(0),
        })
    );
}
