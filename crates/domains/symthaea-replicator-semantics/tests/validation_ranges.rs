// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_replicator_semantics::validation::{
    ResolvedResourceAccountingScheme, ResourceDimensionRule, SchemaValidationError,
};
use symthaea_replicator_semantics::{
    ResourceAccountingSchemeId, ResourceDimensionId, ResourceQuantity, ResourceVector,
};

fn dim(id: u16) -> ResourceDimensionId {
    ResourceDimensionId::new(id)
}

fn quantity(id: u16, amount: u64) -> ResourceQuantity {
    ResourceQuantity::new(dim(id), amount)
}

#[test]
fn nonzero_minimum_and_optional_dimension_semantics_are_explicit() {
    let id = ResourceAccountingSchemeId::new([0x55; 32]);
    let schema = ResolvedResourceAccountingScheme::new(
        id,
        vec![
            ResourceDimensionRule::new(dim(0), true, 5, 100),
            ResourceDimensionRule::new(dim(1), false, 0, 100),
        ],
    )
    .unwrap();

    let required_only = ResourceVector::new(id, vec![quantity(0, 5)]).unwrap();
    assert!(schema.validate(required_only).is_ok());

    let below_minimum = ResourceVector::new(id, vec![quantity(0, 4)]).unwrap();
    assert_eq!(
        schema.validate(below_minimum),
        Err(SchemaValidationError::ResourceAmountBelowMinimum {
            dimension: dim(0),
            amount: 4,
            minimum: 5,
        })
    );

    let with_optional =
        ResourceVector::new(id, vec![quantity(0, 5), quantity(1, 10)]).unwrap();
    assert!(schema.validate(with_optional).is_ok());

    let unknown = ResourceVector::new(id, vec![quantity(0, 5), quantity(2, 1)]).unwrap();
    assert_eq!(
        schema.validate(unknown),
        Err(SchemaValidationError::UnknownResourceDimension { dimension: dim(2) })
    );
}
