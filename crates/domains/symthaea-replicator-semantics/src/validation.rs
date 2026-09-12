// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structural schema validation for schema-bound RSK values.
//!
//! This module does not verify registry provenance, signatures, trust snapshots,
//! policy, grants, or physical-domain semantics. It only proves that one bound
//! value is structurally valid under one resolved schema with the exact bound ID.

use crate::{
    BoundCapabilitySet, CapabilitySchemaId, ResourceAccountingSchemeId, ResourceDimensionId,
    ResourceVector, SemanticBindingError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityBitClass {
    Assignable,
    Reserved,
    Retired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityBitRule {
    bit: u8,
    class: CapabilityBitClass,
}
impl CapabilityBitRule {
    pub const fn new(bit: u8, class: CapabilityBitClass) -> Self {
        Self { bit, class }
    }

    pub const fn bit(self) -> u8 {
        self.bit
    }

    pub const fn class(self) -> CapabilityBitClass {
        self.class
    }
}

/// Structurally resolved capability schema.
///
/// The ID is an externally supplied exact semantic identity. This type validates
/// internal rule-table consistency but does not prove where the schema came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCapabilitySchema {
    id: CapabilitySchemaId,
    bit_width: u8,
    rules: Vec<CapabilityBitRule>,
}
impl ResolvedCapabilitySchema {
    pub fn new(
        id: CapabilitySchemaId,
        bit_width: u8,
        rules: Vec<CapabilityBitRule>,
    ) -> Result<Self, SchemaValidationError> {
        if !(1..=64).contains(&bit_width) {
            return Err(SchemaValidationError::InvalidCapabilityBitWidth { bit_width });
        }
        if rules.windows(2).any(|pair| pair[0].bit >= pair[1].bit) {
            return Err(SchemaValidationError::NonCanonicalCapabilityRules);
        }
        if let Some(rule) = rules.iter().find(|rule| rule.bit >= bit_width) {
            return Err(SchemaValidationError::CapabilityRuleOutsideBitWidth {
                bit: rule.bit,
                bit_width,
            });
        }
        Ok(Self {
            id,
            bit_width,
            rules,
        })
    }

    pub const fn id(&self) -> CapabilitySchemaId {
        self.id
    }

    pub const fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn rules(&self) -> &[CapabilityBitRule] {
        &self.rules
    }

    pub fn validate(
        &self,
        value: BoundCapabilitySet,
    ) -> Result<ValidatedCapabilitySet, SchemaValidationError> {
        if value.schema() != self.id {
            return Err(SchemaValidationError::Binding(
                SemanticBindingError::CapabilitySchemaMismatch {
                    left: value.schema(),
                    right: self.id,
                },
            ));
        }

        let mut bits = value.bits();
        while bits != 0 {
            let bit = bits.trailing_zeros() as u8;
            bits &= bits - 1;

            if bit >= self.bit_width {
                return Err(SchemaValidationError::CapabilityBitOutsideWidth {
                    bit,
                    bit_width: self.bit_width,
                });
            }

            match self.rules.binary_search_by_key(&bit, |rule| rule.bit) {
                Ok(index) => match self.rules[index].class {
                    CapabilityBitClass::Assignable => {}
                    CapabilityBitClass::Reserved => {
                        return Err(SchemaValidationError::ReservedCapabilityBit { bit });
                    }
                    CapabilityBitClass::Retired => {
                        return Err(SchemaValidationError::RetiredCapabilityBit { bit });
                    }
                },
                Err(_) => return Err(SchemaValidationError::UnknownCapabilityBit { bit }),
            }
        }

        Ok(ValidatedCapabilitySet { value })
    }
}

/// Capability value proven structurally valid under its exact resolved schema.
///
/// Fields are private so callers cannot mint this type from arbitrary bound bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValidatedCapabilitySet {
    value: BoundCapabilitySet,
}
impl ValidatedCapabilitySet {
    pub const fn bound(self) -> BoundCapabilitySet {
        self.value
    }

    pub const fn schema(self) -> CapabilitySchemaId {
        self.value.schema()
    }

    pub const fn bits(self) -> u64 {
        self.value.bits()
    }

    pub const fn is_empty(self) -> bool {
        self.value.is_empty()
    }

    pub fn is_subset_of(self, other: Self) -> Result<bool, SchemaValidationError> {
        self.value
            .is_subset_of(other.value)
            .map_err(SchemaValidationError::Binding)
    }

    pub fn intersect(self, other: Self) -> Result<Self, SchemaValidationError> {
        Ok(Self {
            value: self
                .value
                .intersect(other.value)
                .map_err(SchemaValidationError::Binding)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceDimensionRule {
    dimension: ResourceDimensionId,
    required: bool,
    minimum: u64,
    maximum: u64,
}
impl ResourceDimensionRule {
    pub const fn new(
        dimension: ResourceDimensionId,
        required: bool,
        minimum: u64,
        maximum: u64,
    ) -> Self {
        Self {
            dimension,
            required,
            minimum,
            maximum,
        }
    }

    pub const fn dimension(self) -> ResourceDimensionId {
        self.dimension
    }

    pub const fn required(self) -> bool {
        self.required
    }

    pub const fn minimum(self) -> u64 {
        self.minimum
    }

    pub const fn maximum(self) -> u64 {
        self.maximum
    }
}

/// Structurally resolved resource-accounting schema.
///
/// This proves only table consistency and exact scheme binding. Registry/trust
/// provenance is deliberately outside this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedResourceAccountingScheme {
    id: ResourceAccountingSchemeId,
    rules: Vec<ResourceDimensionRule>,
}
impl ResolvedResourceAccountingScheme {
    pub fn new(
        id: ResourceAccountingSchemeId,
        rules: Vec<ResourceDimensionRule>,
    ) -> Result<Self, SchemaValidationError> {
        if rules.is_empty() {
            return Err(SchemaValidationError::EmptyResourceSchema);
        }
        if rules
            .windows(2)
            .any(|pair| pair[0].dimension >= pair[1].dimension)
        {
            return Err(SchemaValidationError::NonCanonicalResourceSchemaDimensions);
        }
        if let Some(rule) = rules.iter().find(|rule| rule.minimum > rule.maximum) {
            return Err(SchemaValidationError::InvalidResourceRange {
                dimension: rule.dimension,
                minimum: rule.minimum,
                maximum: rule.maximum,
            });
        }
        Ok(Self { id, rules })
    }

    pub const fn id(&self) -> ResourceAccountingSchemeId {
        self.id
    }

    pub fn rules(&self) -> &[ResourceDimensionRule] {
        &self.rules
    }

    pub fn validate(
        &self,
        value: ResourceVector,
    ) -> Result<ValidatedResourceVector, SchemaValidationError> {
        if value.scheme() != self.id {
            return Err(SchemaValidationError::Binding(
                SemanticBindingError::ResourceSchemeMismatch {
                    left: value.scheme(),
                    right: self.id,
                },
            ));
        }

        for quantity in value.quantities() {
            let dimension = quantity.dimension();
            let Ok(index) = self
                .rules
                .binary_search_by_key(&dimension, |rule| rule.dimension)
            else {
                return Err(SchemaValidationError::UnknownResourceDimension { dimension });
            };
            let rule = self.rules[index];
            if quantity.amount() < rule.minimum {
                return Err(SchemaValidationError::ResourceAmountBelowMinimum {
                    dimension,
                    amount: quantity.amount(),
                    minimum: rule.minimum,
                });
            }
            if quantity.amount() > rule.maximum {
                return Err(SchemaValidationError::ResourceAmountAboveMaximum {
                    dimension,
                    amount: quantity.amount(),
                    maximum: rule.maximum,
                });
            }
        }

        for rule in self.rules.iter().filter(|rule| rule.required) {
            if value
                .quantities()
                .binary_search_by_key(&rule.dimension, |quantity| quantity.dimension())
                .is_err()
            {
                return Err(SchemaValidationError::MissingRequiredResourceDimension {
                    dimension: rule.dimension,
                });
            }
        }

        Ok(ValidatedResourceVector { value })
    }

    pub fn checked_remaining(
        &self,
        limit: &ValidatedResourceVector,
        consumed: &ValidatedResourceVector,
    ) -> Result<ValidatedResourceVector, SchemaValidationError> {
        self.ensure_validated_scheme(limit)?;
        self.ensure_validated_scheme(consumed)?;
        let remaining = limit
            .value
            .checked_remaining(&consumed.value)
            .map_err(SchemaValidationError::Binding)?;
        self.validate(remaining)
    }

    pub fn checked_add(
        &self,
        left: &ValidatedResourceVector,
        right: &ValidatedResourceVector,
    ) -> Result<ValidatedResourceVector, SchemaValidationError> {
        self.ensure_validated_scheme(left)?;
        self.ensure_validated_scheme(right)?;
        let combined = left
            .value
            .checked_add(&right.value)
            .map_err(SchemaValidationError::Binding)?;
        self.validate(combined)
    }

    pub fn verify_conservative_remaining_transition(
        &self,
        target_remaining: &ValidatedResourceVector,
        conservative_envelope: &ValidatedResourceVector,
    ) -> Result<(), SchemaValidationError> {
        self.ensure_validated_scheme(target_remaining)?;
        self.ensure_validated_scheme(conservative_envelope)?;
        ResourceVector::verify_conservative_remaining_transition(
            &target_remaining.value,
            &conservative_envelope.value,
        )
        .map_err(SchemaValidationError::Binding)
    }

    fn ensure_validated_scheme(
        &self,
        value: &ValidatedResourceVector,
    ) -> Result<(), SchemaValidationError> {
        if value.value.scheme() == self.id {
            Ok(())
        } else {
            Err(SchemaValidationError::Binding(
                SemanticBindingError::ResourceSchemeMismatch {
                    left: value.value.scheme(),
                    right: self.id,
                },
            ))
        }
    }
}

/// Resource vector proven structurally valid under its exact resolved scheme.
///
/// Fields are private so callers cannot bypass required-dimension/range checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedResourceVector {
    value: ResourceVector,
}
impl ValidatedResourceVector {
    pub fn bound(&self) -> &ResourceVector {
        &self.value
    }

    pub const fn scheme(&self) -> ResourceAccountingSchemeId {
        self.value.scheme()
    }

    pub fn quantities(&self) -> &[crate::ResourceQuantity] {
        self.value.quantities()
    }

    pub fn fits_within(&self, ceiling: &Self) -> Result<bool, SchemaValidationError> {
        self.value
            .fits_within(&ceiling.value)
            .map_err(SchemaValidationError::Binding)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaValidationError {
    Binding(SemanticBindingError),
    InvalidCapabilityBitWidth {
        bit_width: u8,
    },
    NonCanonicalCapabilityRules,
    CapabilityRuleOutsideBitWidth {
        bit: u8,
        bit_width: u8,
    },
    CapabilityBitOutsideWidth {
        bit: u8,
        bit_width: u8,
    },
    ReservedCapabilityBit {
        bit: u8,
    },
    RetiredCapabilityBit {
        bit: u8,
    },
    UnknownCapabilityBit {
        bit: u8,
    },
    EmptyResourceSchema,
    NonCanonicalResourceSchemaDimensions,
    InvalidResourceRange {
        dimension: ResourceDimensionId,
        minimum: u64,
        maximum: u64,
    },
    UnknownResourceDimension {
        dimension: ResourceDimensionId,
    },
    MissingRequiredResourceDimension {
        dimension: ResourceDimensionId,
    },
    ResourceAmountBelowMinimum {
        dimension: ResourceDimensionId,
        amount: u64,
        minimum: u64,
    },
    ResourceAmountAboveMaximum {
        dimension: ResourceDimensionId,
        amount: u64,
        maximum: u64,
    },
}

impl From<SemanticBindingError> for SchemaValidationError {
    fn from(value: SemanticBindingError) -> Self {
        Self::Binding(value)
    }
}
