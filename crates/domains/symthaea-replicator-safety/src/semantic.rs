// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Schema-bound authority values for the Replicator Safety Kernel (RSK).
//!
//! These types model semantic identity only. They contain no physical,
//! biological, molecular, manufacturing, or other replication mechanism.

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

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
        let capability = &root["capability_schema"];
        let resource = &root["resource_schema"];
        assert_eq!(
            digest_value(capability),
            root["capability_schema_sha256"].as_str().unwrap()
        );
        assert_eq!(
            digest_value(resource),
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
        let wrong_scheme =
            ResourceVector::new(b, vec![quantity(0, 1), quantity(1, 2)]).unwrap();
        let wrong_dimensions =
            ResourceVector::new(a, vec![quantity(0, 1), quantity(2, 2)]).unwrap();

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
    fn resource_underflow_and_overflow_fail_closed() {
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
    }

    #[test]
    fn conservative_transition_envelope_can_only_reduce_remaining_authority() {
        let scheme = ResourceAccountingSchemeId::new([8; 32]);
        let envelope =
            ResourceVector::new(scheme, vec![quantity(0, 5), quantity(1, 7)]).unwrap();
        let narrower =
            ResourceVector::new(scheme, vec![quantity(0, 4), quantity(1, 7)]).unwrap();
        let wider = ResourceVector::new(scheme, vec![quantity(0, 6), quantity(1, 7)]).unwrap();

        ResourceVector::verify_conservative_remaining_transition(&narrower, &envelope).unwrap();
        assert_eq!(
            ResourceVector::verify_conservative_remaining_transition(&wider, &envelope),
            Err(SemanticBindingError::TransitionExceedsConservativeEnvelope {
                dimension: ResourceDimensionId::new(0),
            })
        );
    }
}
