// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Schema-bound authority values for the Replicator Safety Kernel (RSK).
//!
//! These types model semantic identity only. They contain no physical,
//! biological, molecular, manufacturing, or other replication mechanism.

#![forbid(unsafe_code)]

/// Frozen identity of one capability vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilitySchemaId([u8; 32]);
impl CapabilitySchemaId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// A capability set whose numeric bits are meaningful only under `schema`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoundCapabilitySet {
    schema: CapabilitySchemaId,
    bits: u64,
}
impl BoundCapabilitySet {
    pub const fn new(schema: CapabilitySchemaId, bits: u64) -> Self {
        Self { schema, bits }
    }

    pub const fn none(schema: CapabilitySchemaId) -> Self {
        Self { schema, bits: 0 }
    }

    pub const fn schema(self) -> CapabilitySchemaId {
        self.schema
    }

    pub const fn bits(self) -> u64 {
        self.bits
    }

    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    pub fn is_subset_of(self, other: Self) -> Result<bool, SemanticBindingError> {
        ensure_capability_schema(self.schema, other.schema)?;
        Ok(self.bits & !other.bits == 0)
    }

    pub fn intersect(self, other: Self) -> Result<Self, SemanticBindingError> {
        ensure_capability_schema(self.schema, other.schema)?;
        Ok(Self::new(self.schema, self.bits & other.bits))
    }
}

/// Frozen identity of one resource-accounting vocabulary and arithmetic policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceAccountingSchemeId([u8; 32]);
impl ResourceAccountingSchemeId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Stable numeric identifier for one dimension *within* an accounting scheme.
///
/// The identifier has no cross-scheme meaning by itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceDimensionId(u16);
impl ResourceDimensionId {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceQuantity {
    dimension: ResourceDimensionId,
    amount: u64,
}
impl ResourceQuantity {
    pub const fn new(dimension: ResourceDimensionId, amount: u64) -> Self {
        Self { dimension, amount }
    }

    pub const fn dimension(self) -> ResourceDimensionId {
        self.dimension
    }

    pub const fn amount(self) -> u64 {
        self.amount
    }
}

/// Canonical resource vector under one exact accounting scheme.
///
/// Quantities must be non-empty and strictly increasing by numeric dimension
/// identifier. Arithmetic is only defined for vectors with identical scheme and
/// dimension identity/order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceVector {
    scheme: ResourceAccountingSchemeId,
    quantities: Vec<ResourceQuantity>,
}
impl ResourceVector {
    pub fn new(
        scheme: ResourceAccountingSchemeId,
        quantities: Vec<ResourceQuantity>,
    ) -> Result<Self, SemanticBindingError> {
        if quantities.is_empty() {
            return Err(SemanticBindingError::EmptyResourceVector);
        }
        if quantities
            .windows(2)
            .any(|pair| pair[0].dimension >= pair[1].dimension)
        {
            return Err(SemanticBindingError::NonCanonicalResourceDimensions);
        }
        Ok(Self { scheme, quantities })
    }

    pub const fn scheme(&self) -> ResourceAccountingSchemeId {
        self.scheme
    }

    pub fn quantities(&self) -> &[ResourceQuantity] {
        &self.quantities
    }

    pub fn checked_remaining(
        &self,
        consumed: &Self,
    ) -> Result<Self, SemanticBindingError> {
        self.ensure_compatible(consumed)?;
        let mut remaining = Vec::with_capacity(self.quantities.len());
        for (limit, used) in self.quantities.iter().zip(&consumed.quantities) {
            let Some(amount) = limit.amount.checked_sub(used.amount) else {
                return Err(SemanticBindingError::ResourceUnderflow {
                    dimension: limit.dimension,
                });
            };
            remaining.push(ResourceQuantity::new(limit.dimension, amount));
        }
        Ok(Self {
            scheme: self.scheme,
            quantities: remaining,
        })
    }

    pub fn checked_add(&self, other: &Self) -> Result<Self, SemanticBindingError> {
        self.ensure_compatible(other)?;
        let mut combined = Vec::with_capacity(self.quantities.len());
        for (left, right) in self.quantities.iter().zip(&other.quantities) {
            let Some(amount) = left.amount.checked_add(right.amount) else {
                return Err(SemanticBindingError::ResourceOverflow {
                    dimension: left.dimension,
                });
            };
            combined.push(ResourceQuantity::new(left.dimension, amount));
        }
        Ok(Self {
            scheme: self.scheme,
            quantities: combined,
        })
    }

    pub fn fits_within(&self, ceiling: &Self) -> Result<bool, SemanticBindingError> {
        self.ensure_compatible(ceiling)?;
        Ok(self
            .quantities
            .iter()
            .zip(&ceiling.quantities)
            .all(|(value, limit)| value.amount <= limit.amount))
    }

    /// Verify only the structural attenuation claim supplied by a separately
    /// trusted migration process.
    ///
    /// This does **not** infer the conservative image of a source schema. It
    /// merely checks that `target_remaining` is no larger than an already
    /// supplied target-schema envelope.
    pub fn verify_conservative_remaining_transition(
        target_remaining: &Self,
        conservative_envelope: &Self,
    ) -> Result<(), SemanticBindingError> {
        target_remaining.ensure_compatible(conservative_envelope)?;
        for (target, envelope) in target_remaining
            .quantities
            .iter()
            .zip(&conservative_envelope.quantities)
        {
            if target.amount > envelope.amount {
                return Err(SemanticBindingError::TransitionExceedsConservativeEnvelope {
                    dimension: target.dimension,
                });
            }
        }
        Ok(())
    }

    fn ensure_compatible(&self, other: &Self) -> Result<(), SemanticBindingError> {
        if self.scheme != other.scheme {
            return Err(SemanticBindingError::ResourceSchemeMismatch {
                left: self.scheme,
                right: other.scheme,
            });
        }
        if self.quantities.len() != other.quantities.len()
            || self
                .quantities
                .iter()
                .zip(&other.quantities)
                .any(|(left, right)| left.dimension != right.dimension)
        {
            return Err(SemanticBindingError::ResourceDimensionSetMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticBindingError {
    CapabilitySchemaMismatch {
        left: CapabilitySchemaId,
        right: CapabilitySchemaId,
    },
    ResourceSchemeMismatch {
        left: ResourceAccountingSchemeId,
        right: ResourceAccountingSchemeId,
    },
    EmptyResourceVector,
    NonCanonicalResourceDimensions,
    ResourceDimensionSetMismatch,
    ResourceUnderflow {
        dimension: ResourceDimensionId,
    },
    ResourceOverflow {
        dimension: ResourceDimensionId,
    },
    TransitionExceedsConservativeEnvelope {
        dimension: ResourceDimensionId,
    },
}

fn ensure_capability_schema(
    left: CapabilitySchemaId,
    right: CapabilitySchemaId,
) -> Result<(), SemanticBindingError> {
    if left == right {
        Ok(())
    } else {
        Err(SemanticBindingError::CapabilitySchemaMismatch { left, right })
    }
}
