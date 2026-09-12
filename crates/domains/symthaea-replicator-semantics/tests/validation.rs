// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_replicator_semantics::validation::{
    CapabilityBitClass, CapabilityBitRule, ResolvedCapabilitySchema,
    ResolvedResourceAccountingScheme, ResourceDimensionRule, SchemaValidationError,
};
use symthaea_replicator_semantics::{
    BoundCapabilitySet, CapabilitySchemaId, ResourceAccountingSchemeId, ResourceDimensionId,
    ResourceQuantity, ResourceVector, SemanticBindingError,
};

fn capability_schema_id() -> CapabilitySchemaId {
    CapabilitySchemaId::new([
        0xda, 0x00, 0x4c, 0x77, 0xda, 0x0d, 0xf5, 0x12, 0xef, 0x77, 0x2a, 0xa1, 0x67,
        0xfd, 0x38, 0x6b, 0x61, 0xd6, 0xa5, 0x81, 0xa3, 0xbe, 0x40, 0xde, 0xd9, 0x3d,
        0x05, 0x6e, 0x36, 0xdb, 0xc8, 0x56,
    ])
}

fn resource_schema_id() -> ResourceAccountingSchemeId {
    ResourceAccountingSchemeId::new([
        0xd0, 0x55, 0xf5, 0xe3, 0x93, 0x74, 0x73, 0x80, 0x0e, 0x21, 0x6c, 0xfa, 0x44,
        0x2d, 0xa2, 0xe6, 0x1d, 0xd9, 0x86, 0x24, 0x35, 0x7e, 0xa6, 0xab, 0xed, 0x3d,
        0x2e, 0x7a, 0x90, 0xc8, 0xe0, 0x3a,
    ])
}

fn golden_capability_schema() -> ResolvedCapabilitySchema {
    ResolvedCapabilitySchema::new(
        capability_schema_id(),
        8,
        vec![
            CapabilityBitRule::new(0, CapabilityBitClass::Assignable),
            CapabilityBitRule::new(1, CapabilityBitClass::Assignable),
            CapabilityBitRule::new(2, CapabilityBitClass::Reserved),
            CapabilityBitRule::new(3, CapabilityBitClass::Retired),
            CapabilityBitRule::new(7, CapabilityBitClass::Reserved),
        ],
    )
    .unwrap()
}

fn dim(id: u16) -> ResourceDimensionId {
    ResourceDimensionId::new(id)
}

fn quantity(id: u16, amount: u64) -> ResourceQuantity {
    ResourceQuantity::new(dim(id), amount)
}

fn golden_resource_schema() -> ResolvedResourceAccountingScheme {
    ResolvedResourceAccountingScheme::new(
        resource_schema_id(),
        vec![
            ResourceDimensionRule::new(dim(0), true, 0, 1_000),
            ResourceDimensionRule::new(dim(1), true, 0, 1_000),
        ],
    )
    .unwrap()
}

#[test]
fn malformed_capability_schema_tables_fail_closed() {
    let id = CapabilitySchemaId::new([1; 32]);
    assert_eq!(
        ResolvedCapabilitySchema::new(id, 0, vec![]),
        Err(SchemaValidationError::InvalidCapabilityBitWidth { bit_width: 0 })
    );
    assert_eq!(
        ResolvedCapabilitySchema::new(
            id,
            8,
            vec![
                CapabilityBitRule::new(1, CapabilityBitClass::Assignable),
                CapabilityBitRule::new(1, CapabilityBitClass::Reserved),
            ],
        ),
        Err(SchemaValidationError::NonCanonicalCapabilityRules)
    );
    assert_eq!(
        ResolvedCapabilitySchema::new(
            id,
            8,
            vec![CapabilityBitRule::new(8, CapabilityBitClass::Assignable)],
        ),
        Err(SchemaValidationError::CapabilityRuleOutsideBitWidth {
            bit: 8,
            bit_width: 8,
        })
    );
}

#[test]
fn only_assignable_capability_bits_validate() {
    let schema = golden_capability_schema();
    let valid = schema
        .validate(BoundCapabilitySet::new(capability_schema_id(), 0b11))
        .unwrap();
    assert_eq!(valid.bits(), 0b11);

    assert_eq!(
        schema.validate(BoundCapabilitySet::new(capability_schema_id(), 1 << 2)),
        Err(SchemaValidationError::ReservedCapabilityBit { bit: 2 })
    );
    assert_eq!(
        schema.validate(BoundCapabilitySet::new(capability_schema_id(), 1 << 3)),
        Err(SchemaValidationError::RetiredCapabilityBit { bit: 3 })
    );
    assert_eq!(
        schema.validate(BoundCapabilitySet::new(capability_schema_id(), 1 << 4)),
        Err(SchemaValidationError::UnknownCapabilityBit { bit: 4 })
    );
    assert_eq!(
        schema.validate(BoundCapabilitySet::new(capability_schema_id(), 1 << 8)),
        Err(SchemaValidationError::CapabilityBitOutsideWidth {
            bit: 8,
            bit_width: 8,
        })
    );
}

#[test]
fn capability_validation_requires_exact_schema_identity() {
    let schema = golden_capability_schema();
    let wrong = CapabilitySchemaId::new([9; 32]);
    assert_eq!(
        schema.validate(BoundCapabilitySet::new(wrong, 0b01)),
        Err(SchemaValidationError::Binding(
            SemanticBindingError::CapabilitySchemaMismatch {
                left: wrong,
                right: capability_schema_id(),
            }
        ))
    );
}

#[test]
fn validated_capability_operations_preserve_type_state() {
    let schema = golden_capability_schema();
    let broad = schema
        .validate(BoundCapabilitySet::new(capability_schema_id(), 0b11))
        .unwrap();
    let narrow = schema
        .validate(BoundCapabilitySet::new(capability_schema_id(), 0b01))
        .unwrap();
    assert!(narrow.is_subset_of(broad).unwrap());
    assert_eq!(broad.intersect(narrow).unwrap().bits(), 0b01);
}

#[test]
fn malformed_resource_schema_tables_fail_closed() {
    let id = ResourceAccountingSchemeId::new([2; 32]);
    assert_eq!(
        ResolvedResourceAccountingScheme::new(id, vec![]),
        Err(SchemaValidationError::EmptyResourceSchema)
    );
    assert_eq!(
        ResolvedResourceAccountingScheme::new(
            id,
            vec![
                ResourceDimensionRule::new(dim(1), true, 0, 10),
                ResourceDimensionRule::new(dim(1), true, 0, 10),
            ],
        ),
        Err(SchemaValidationError::NonCanonicalResourceSchemaDimensions)
    );
    assert_eq!(
        ResolvedResourceAccountingScheme::new(
            id,
            vec![ResourceDimensionRule::new(dim(0), true, 11, 10)],
        ),
        Err(SchemaValidationError::InvalidResourceRange {
            dimension: dim(0),
            minimum: 11,
            maximum: 10,
        })
    );
}

#[test]
fn resource_validation_requires_known_required_dimensions_and_ranges() {
    let schema = golden_resource_schema();
    let valid = ResourceVector::new(
        resource_schema_id(),
        vec![quantity(0, 100), quantity(1, 80)],
    )
    .unwrap();
    assert_eq!(schema.validate(valid).unwrap().quantities().len(), 2);

    let missing = ResourceVector::new(resource_schema_id(), vec![quantity(0, 10)]).unwrap();
    assert_eq!(
        schema.validate(missing),
        Err(SchemaValidationError::MissingRequiredResourceDimension { dimension: dim(1) })
    );

    let unknown = ResourceVector::new(
        resource_schema_id(),
        vec![quantity(0, 10), quantity(1, 10), quantity(2, 1)],
    )
    .unwrap();
    assert_eq!(
        schema.validate(unknown),
        Err(SchemaValidationError::UnknownResourceDimension { dimension: dim(2) })
    );

    let too_large = ResourceVector::new(
        resource_schema_id(),
        vec![quantity(0, 1_001), quantity(1, 10)],
    )
    .unwrap();
    assert_eq!(
        schema.validate(too_large),
        Err(SchemaValidationError::ResourceAmountAboveMaximum {
            dimension: dim(0),
            amount: 1_001,
            maximum: 1_000,
        })
    );
}

#[test]
fn resource_validation_requires_exact_scheme_identity() {
    let schema = golden_resource_schema();
    let wrong = ResourceAccountingSchemeId::new([8; 32]);
    let value = ResourceVector::new(wrong, vec![quantity(0, 10), quantity(1, 10)]).unwrap();
    assert_eq!(
        schema.validate(value),
        Err(SchemaValidationError::Binding(
            SemanticBindingError::ResourceSchemeMismatch {
                left: wrong,
                right: resource_schema_id(),
            }
        ))
    );
}

#[test]
fn validated_resource_arithmetic_is_revalidated() {
    let schema = golden_resource_schema();
    let limit = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 100), quantity(1, 80)],
            )
            .unwrap(),
        )
        .unwrap();
    let consumed = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 40), quantity(1, 20)],
            )
            .unwrap(),
        )
        .unwrap();
    let remaining = schema.checked_remaining(&limit, &consumed).unwrap();
    assert_eq!(
        remaining.quantities(),
        &[quantity(0, 60), quantity(1, 60)]
    );

    let almost_max = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 900), quantity(1, 20)],
            )
            .unwrap(),
        )
        .unwrap();
    let addend = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 200), quantity(1, 20)],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        schema.checked_add(&almost_max, &addend),
        Err(SchemaValidationError::ResourceAmountAboveMaximum {
            dimension: dim(0),
            amount: 1_100,
            maximum: 1_000,
        })
    );
}

#[test]
fn conservative_transition_cannot_widen_validated_remaining_authority() {
    let schema = golden_resource_schema();
    let envelope = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 60), quantity(1, 60)],
            )
            .unwrap(),
        )
        .unwrap();
    let narrower = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 59), quantity(1, 60)],
            )
            .unwrap(),
        )
        .unwrap();
    let wider = schema
        .validate(
            ResourceVector::new(
                resource_schema_id(),
                vec![quantity(0, 61), quantity(1, 60)],
            )
            .unwrap(),
        )
        .unwrap();

    schema
        .verify_conservative_remaining_transition(&narrower, &envelope)
        .unwrap();
    assert_eq!(
        schema.verify_conservative_remaining_transition(&wider, &envelope),
        Err(SchemaValidationError::Binding(
            SemanticBindingError::TransitionExceedsConservativeEnvelope {
                dimension: dim(0),
            }
        ))
    );
}
