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

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use symthaea_evidence_plane::ContentAddress32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalLineageError {
    EmptyEvidence,
}

impl fmt::Display for ModalLineageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEvidence => write!(f, "modal lineage requires at least one evidence reference"),
        }
    }
}

impl std::error::Error for ModalLineageError {}

/// Exact content lineage for one modal representation.
///
/// `evidence` is canonicalized as an order-independent set because reordering
/// independent evidence references must not create a different lineage receipt.
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
    pub fn with_transform(mut self, transform: ContentAddress32) -> Self {
        self.transforms.push(transform);
        self
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
    evidence: Vec<ContentAddress32>,
    #[serde(default)]
    input_space: Option<ContentAddress32>,
    #[serde(default)]
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
        receipt.input_space = wire.input_space;
        receipt.transforms = wire.transforms;
        receipt.output_space = wire.output_space;
        Ok(receipt)
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
    fn transform_order_is_semantic() {
        let evidence = address("symthaea-evidence-v1", 1);
        let transform_a = address("symthaea-transform-a-v1", 2);
        let transform_b = address("symthaea-transform-b-v1", 3);

        let ab = ModalLineageReceipt::from_single_evidence(evidence.clone())
            .with_transform(transform_a.clone())
            .with_transform(transform_b.clone());
        let ba = ModalLineageReceipt::from_single_evidence(evidence)
            .with_transform(transform_b)
            .with_transform(transform_a);

        assert_ne!(ab, ba);
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
        .with_output_space(address("symthaea-output-space-v1", 5));

        let encoded = serde_json::to_string(&receipt).unwrap();
        let decoded: ModalLineageReceipt = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, receipt);
    }

    #[test]
    fn serde_rejects_empty_evidence_and_unknown_fields() {
        let empty = serde_json::json!({
            "evidence": [],
            "input_space": null,
            "transforms": [],
            "output_space": null
        });
        assert!(serde_json::from_value::<ModalLineageReceipt>(empty).is_err());

        let unknown = serde_json::json!({
            "evidence": [{
                "algorithm": "blake3-256",
                "namespace": "symthaea-evidence-v1",
                "digest": [1; 32]
            }],
            "input_space": null,
            "transforms": [],
            "output_space": null,
            "trust": 1.0
        });
        assert!(serde_json::from_value::<ModalLineageReceipt>(unknown).is_err());
    }
}
