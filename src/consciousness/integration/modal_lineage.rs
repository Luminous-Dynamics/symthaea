// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed lineage for a modal representation crossing into consciousness.
//!
//! This is deliberately complementary to `mind::provenance::ProvenanceTag`:
//!
//! - provenance answers **which causal source/process produced information?**
//! - lineage answers **which exact evidence/representation/transform identities
//!   produced this modal representation?**
//!
//! A lineage receipt is content identity only. It does not imply authenticity,
//! trust, epistemic rank, sensor validity, or causal truth. Those remain separate
//! contracts.
//!
//! Direct receipt fan-in is intentionally bounded. Large evidence collections
//! should cross this boundary as one domain-owned, content-addressed bundle rather
//! than as an unbounded list of leaf references. This keeps the generic modal
//! boundary compact while allowing domains to retain arbitrarily rich evidence
//! behind the referenced bundle identity.

use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, SeqAccess, Visitor},
};
use symthaea_evidence_plane::ContentAddress32;

/// Maximum number of direct evidence references carried by one modal receipt.
///
/// Larger evidence sets should be summarized by a domain-owned content-addressed
/// bundle and referenced here as a single evidence identity.
pub const MAX_MODAL_LINEAGE_EVIDENCE_REFERENCES: usize = 256;

/// Maximum number of ordered transform/policy identities in one modal receipt.
pub const MAX_MODAL_LINEAGE_TRANSFORMS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalLineageError {
    EmptyEvidence,
    TooManyEvidenceReferences { actual: usize, max: usize },
    TooManyTransforms { actual: usize, max: usize },
}

impl fmt::Display for ModalLineageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEvidence => {
                write!(f, "modal lineage requires at least one evidence reference")
            }
            Self::TooManyEvidenceReferences { actual, max } => write!(
                f,
                "modal lineage contains {actual} direct evidence references; maximum is {max}"
            ),
            Self::TooManyTransforms { actual, max } => write!(
                f,
                "modal lineage contains {actual} transform references; maximum is {max}"
            ),
        }
    }
}

impl std::error::Error for ModalLineageError {}

/// Exact content lineage for one modal representation.
///
/// `evidence` is canonicalized as an order-independent set because reordering
/// independent evidence references must not create a different lineage receipt.
/// Duplicate references still count against the inbound structural limit before
/// canonicalization so repetition cannot bypass the wire bound.
///
/// `transforms`, by contrast, is intentionally ordered: applying transform A then
/// B is not assumed equivalent to applying B then A.
///
/// Representation-space identities are optional so the same contract can cover
/// both space-aware HDC paths and modalities that only have an evidence identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModalLineageReceipt {
    evidence: Vec<ContentAddress32>,
    input_space: Option<ContentAddress32>,
    transforms: Vec<ContentAddress32>,
    output_space: Option<ContentAddress32>,
}

impl ModalLineageReceipt {
    pub fn new(evidence: Vec<ContentAddress32>) -> Result<Self, ModalLineageError> {
        if evidence.is_empty() {
            return Err(ModalLineageError::EmptyEvidence);
        }
        if evidence.len() > MAX_MODAL_LINEAGE_EVIDENCE_REFERENCES {
            return Err(ModalLineageError::TooManyEvidenceReferences {
                actual: evidence.len(),
                max: MAX_MODAL_LINEAGE_EVIDENCE_REFERENCES,
            });
        }

        Ok(Self {
            evidence: canonicalize_evidence(evidence),
            input_space: None,
            transforms: Vec::new(),
            output_space: None,
        })
    }

    pub fn from_single_evidence(evidence: ContentAddress32) -> Self {
        Self {
            evidence: vec![evidence],
            input_space: None,
            transforms: Vec::new(),
            output_space: None,
        }
    }

    pub fn with_input_space(mut self, space: ContentAddress32) -> Self {
        self.input_space = Some(space);
        self
    }

    /// Append one transformation/policy identity in execution order.
    pub fn with_transform(
        mut self,
        transform: ContentAddress32,
    ) -> Result<Self, ModalLineageError> {
        if self.transforms.len() >= MAX_MODAL_LINEAGE_TRANSFORMS {
            return Err(ModalLineageError::TooManyTransforms {
                actual: self.transforms.len() + 1,
                max: MAX_MODAL_LINEAGE_TRANSFORMS,
            });
        }
        self.transforms.push(transform);
        Ok(self)
    }

    pub fn with_output_space(mut self, space: ContentAddress32) -> Self {
        self.output_space = Some(space);
        self
    }

    pub fn evidence(&self) -> &[ContentAddress32] {
        &self.evidence
    }

    pub fn input_space(&self) -> Option<&ContentAddress32> {
        self.input_space.as_ref()
    }

    pub fn transforms(&self) -> &[ContentAddress32] {
        &self.transforms
    }

    pub fn output_space(&self) -> Option<&ContentAddress32> {
        self.output_space.as_ref()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModalLineageWire {
    #[serde(deserialize_with = "deserialize_evidence_references")]
    evidence: Vec<ContentAddress32>,
    #[serde(default)]
    input_space: Option<ContentAddress32>,
    #[serde(default, deserialize_with = "deserialize_transform_references")]
    transforms: Vec<ContentAddress32>,
    #[serde(default)]
    output_space: Option<ContentAddress32>,
}

impl<'de> Deserialize<'de> for ModalLineageReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModalLineageWire::deserialize(deserializer)?;
        let mut receipt = Self::new(wire.evidence).map_err(de::Error::custom)?;
        if wire.transforms.len() > MAX_MODAL_LINEAGE_TRANSFORMS {
            return Err(de::Error::custom(ModalLineageError::TooManyTransforms {
                actual: wire.transforms.len(),
                max: MAX_MODAL_LINEAGE_TRANSFORMS,
            }));
        }
        receipt.input_space = wire.input_space;
        receipt.transforms = wire.transforms;
        receipt.output_space = wire.output_space;
        Ok(receipt)
    }
}

fn deserialize_evidence_references<'de, D>(
    deserializer: D,
) -> Result<Vec<ContentAddress32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_references::<D, MAX_MODAL_LINEAGE_EVIDENCE_REFERENCES>(
        deserializer,
        "evidence",
    )
}

fn deserialize_transform_references<'de, D>(
    deserializer: D,
) -> Result<Vec<ContentAddress32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_references::<D, MAX_MODAL_LINEAGE_TRANSFORMS>(
        deserializer,
        "transforms",
    )
}

fn deserialize_bounded_references<'de, D, const MAX: usize>(
    deserializer: D,
    field: &'static str,
) -> Result<Vec<ContentAddress32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(BoundedReferenceVisitor::<MAX> { field })
}

struct BoundedReferenceVisitor<const MAX: usize> {
    field: &'static str,
}

impl<'de, const MAX: usize> Visitor<'de> for BoundedReferenceVisitor<MAX> {
    type Value = Vec<ContentAddress32>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "at most {MAX} modal-lineage {} references",
            self.field
        )
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if let Some(hint) = seq.size_hint() {
            if hint > MAX {
                return Err(de::Error::custom(format_args!(
                    "modal lineage {} sequence length {hint} exceeds maximum {MAX}",
                    self.field
                )));
            }
        }

        let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(MAX));
        while let Some(value) = seq.next_element()? {
            if values.len() >= MAX {
                return Err(de::Error::custom(format_args!(
                    "modal lineage {} sequence exceeds maximum {MAX}",
                    self.field
                )));
            }
            values.push(value);
        }
        Ok(values)
    }
}

fn canonicalize_evidence(mut evidence: Vec<ContentAddress32>) -> Vec<ContentAddress32> {
    evidence.sort_by_cached_key(ContentAddress32::to_canonical_string);
    evidence.dedup();
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(namespace: &str, byte: u8) -> ContentAddress32 {
        ContentAddress32::new("blake3-256", namespace, [byte; 32]).unwrap()
    }

    #[test]
    fn empty_evidence_is_rejected() {
        assert_eq!(
            ModalLineageReceipt::new(Vec::new()),
            Err(ModalLineageError::EmptyEvidence)
        );
    }

    #[test]
    fn evidence_is_order_independent_and_deduplicated() {
        let a = address("symthaea-evidence-a-v1", 1);
        let b = address("symthaea-evidence-b-v1", 2);

        let left = ModalLineageReceipt::new(vec![a.clone(), b.clone(), a.clone()]).unwrap();
        let right = ModalLineageReceipt::new(vec![b, a]).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.evidence().len(), 2);
    }

    #[test]
    fn duplicate_evidence_cannot_bypass_inbound_reference_bound() {
        let repeated = vec![
            address("symthaea-evidence-v1", 1);
            MAX_MODAL_LINEAGE_EVIDENCE_REFERENCES + 1
        ];
        assert!(matches!(
            ModalLineageReceipt::new(repeated),
            Err(ModalLineageError::TooManyEvidenceReferences { .. })
        ));
    }

    #[test]
    fn transform_order_is_semantic() {
        let evidence = address("symthaea-evidence-v1", 1);
        let transform_a = address("symthaea-transform-a-v1", 2);
        let transform_b = address("symthaea-transform-b-v1", 3);

        let ab = ModalLineageReceipt::from_single_evidence(evidence.clone())
            .with_transform(transform_a.clone())
            .unwrap()
            .with_transform(transform_b.clone())
            .unwrap();
        let ba = ModalLineageReceipt::from_single_evidence(evidence)
            .with_transform(transform_b)
            .unwrap()
            .with_transform(transform_a)
            .unwrap();

        assert_ne!(ab, ba);
    }

    #[test]
    fn programmatic_transform_bound_is_enforced() {
        let mut receipt =
            ModalLineageReceipt::from_single_evidence(address("symthaea-evidence-v1", 1));
        for index in 0..MAX_MODAL_LINEAGE_TRANSFORMS {
            receipt = receipt
                .with_transform(address("symthaea-transform-v1", index as u8))
                .unwrap();
        }
        assert!(matches!(
            receipt.with_transform(address("symthaea-transform-v1", 255)),
            Err(ModalLineageError::TooManyTransforms { .. })
        ));
    }

    #[test]
    fn representation_spaces_remain_distinct_from_transform_identity() {
        let evidence = address("symthaea-evidence-v1", 1);
        let continuous = address("symthaea-continuous-space-v1", 2);
        let projection = address("symthaea-projection-policy-v1", 3);
        let binary = address("symthaea-binary-space-v1", 4);

        let receipt = ModalLineageReceipt::from_single_evidence(evidence)
            .with_input_space(continuous.clone())
            .with_transform(projection.clone())
            .unwrap()
            .with_output_space(binary.clone());

        assert_eq!(receipt.input_space(), Some(&continuous));
        assert_eq!(receipt.transforms(), &[projection]);
        assert_eq!(receipt.output_space(), Some(&binary));
    }

    #[test]
    fn serde_round_trip_preserves_canonical_lineage() {
        let receipt = ModalLineageReceipt::new(vec![
            address("symthaea-evidence-b-v1", 2),
            address("symthaea-evidence-a-v1", 1),
        ])
        .unwrap()
        .with_input_space(address("symthaea-input-space-v1", 3))
        .with_transform(address("symthaea-transform-v1", 4))
        .unwrap()
        .with_output_space(address("symthaea-output-space-v1", 5));

        let encoded = serde_json::to_string(&receipt).unwrap();
        let decoded: ModalLineageReceipt = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, receipt);
    }

    #[test]
    fn serde_rejects_empty_evidence_unknown_fields_and_oversized_sequences() {
        let empty = serde_json::json!({
            "evidence": [],
            "input_space": null,
            "transforms": [],
            "output_space": null
        });
        assert!(serde_json::from_value::<ModalLineageReceipt>(empty).is_err());

        let valid = ModalLineageReceipt::from_single_evidence(address("symthaea-evidence-v1", 1));
        let mut unknown = serde_json::to_value(valid).unwrap();
        unknown["trust"] = serde_json::Value::from(1.0);
        assert!(serde_json::from_value::<ModalLineageReceipt>(unknown).is_err());

        let oversized_evidence = serde_json::json!({
            "evidence": vec![
                address("symthaea-evidence-v1", 1);
                MAX_MODAL_LINEAGE_EVIDENCE_REFERENCES + 1
            ],
            "input_space": null,
            "transforms": [],
            "output_space": null
        });
        assert!(
            serde_json::from_value::<ModalLineageReceipt>(oversized_evidence).is_err()
        );

        let oversized_transforms = serde_json::json!({
            "evidence": [address("symthaea-evidence-v1", 1)],
            "input_space": null,
            "transforms": vec![
                address("symthaea-transform-v1", 2);
                MAX_MODAL_LINEAGE_TRANSFORMS + 1
            ],
            "output_space": null
        });
        assert!(
            serde_json::from_value::<ModalLineageReceipt>(oversized_transforms).is_err()
        );
    }
}
