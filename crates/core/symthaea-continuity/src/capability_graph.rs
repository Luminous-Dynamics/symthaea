// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Closed-world canonical capability graph snapshots.
//!
//! Validation establishes structural closure and deterministic identity only.
//! It does not establish local realization, current availability, verified
//! sufficiency, safety, or authority.
//!
//! Cycles are valid. SCC and recovery-frontier analysis intentionally belong to
//! a later layer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capability::{
    CAPABILITY_DEFINITION_SCHEMA_V1, CapabilityDefinitionId, CapabilityDefinitionV1,
    CapabilityError, CapabilityId,
};

/// Stable schema for a closed capability graph snapshot.
pub const CAPABILITY_GRAPH_SNAPSHOT_SCHEMA_V1: &str =
    "symthaea-continuity-capability-graph-snapshot-v1";

const GRAPH_SNAPSHOT_DOMAIN: &[u8] = b"symthaea.continuity.capability-graph-snapshot.v1\0";

/// Exact content identity of one canonical capability graph snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapabilityGraphSnapshotId([u8; 32]);

impl CapabilityGraphSnapshotId {
    /// Raw BLAKE3-256 identity bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable candidate for a closed capability graph.
///
/// Definitions are canonically ordered by stable [`CapabilityId`]. Every
/// capability reference must resolve to exactly one definition in the same
/// snapshot. Cycles are explicitly permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityGraphSnapshotV1 {
    schema_version: String,
    definitions: Vec<CapabilityDefinitionV1>,
    snapshot_id: CapabilityGraphSnapshotId,
}

impl CapabilityGraphSnapshotV1 {
    /// Construct a canonical, structurally closed graph snapshot.
    pub fn new(
        mut definitions: Vec<CapabilityDefinitionV1>,
    ) -> Result<Self, CapabilityGraphError> {
        for definition in &definitions {
            definition.validate()?;
        }

        definitions.sort_by_key(CapabilityDefinitionV1::id);
        validate_unique_definitions(&definitions)?;
        validate_closed_world(&definitions)?;

        let snapshot_id = CapabilityGraphSnapshotId(hash_snapshot(&definitions));
        Ok(Self {
            schema_version: CAPABILITY_GRAPH_SNAPSHOT_SCHEMA_V1.to_owned(),
            definitions,
            snapshot_id,
        })
    }

    /// Exact content identity of this graph snapshot.
    pub fn id(&self) -> CapabilityGraphSnapshotId {
        self.snapshot_id
    }

    /// Canonically ordered capability definitions.
    pub fn definitions(&self) -> &[CapabilityDefinitionV1] {
        &self.definitions
    }

    /// Revalidate an untrusted/transported snapshot and return a non-Serde
    /// downstream-trusted wrapper.
    pub fn validate(&self) -> Result<ValidatedCapabilityGraphV1, CapabilityGraphError> {
        if self.schema_version != CAPABILITY_GRAPH_SNAPSHOT_SCHEMA_V1 {
            return Err(CapabilityGraphError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }

        for definition in &self.definitions {
            definition.validate()?;
        }

        if self.definitions.windows(2).any(|pair| pair[0].id() >= pair[1].id()) {
            if self
                .definitions
                .windows(2)
                .any(|pair| pair[0].id() == pair[1].id())
            {
                return Err(CapabilityGraphError::DuplicateCapabilityDefinition);
            }
            return Err(CapabilityGraphError::NonCanonicalDefinitionOrder);
        }

        validate_closed_world(&self.definitions)?;

        let expected = CapabilityGraphSnapshotId(hash_snapshot(&self.definitions));
        if expected != self.snapshot_id {
            return Err(CapabilityGraphError::SnapshotIdentityMismatch);
        }

        let index = self
            .definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| (definition.id(), index))
            .collect();

        Ok(ValidatedCapabilityGraphV1 {
            inner: self.clone(),
            index,
        })
    }
}

/// Non-Serde validated graph wrapper.
///
/// Construction is possible only through [`CapabilityGraphSnapshotV1::validate`].
/// The wrapper proves canonical structure and closed references, not operational
/// availability or authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCapabilityGraphV1 {
    inner: CapabilityGraphSnapshotV1,
    index: BTreeMap<CapabilityId, usize>,
}

impl ValidatedCapabilityGraphV1 {
    /// Exact snapshot identity.
    pub fn id(&self) -> CapabilityGraphSnapshotId {
        self.inner.id()
    }

    /// Number of capability definitions in this snapshot.
    pub fn len(&self) -> usize {
        self.inner.definitions.len()
    }

    /// Whether this snapshot has no capability definitions.
    pub fn is_empty(&self) -> bool {
        self.inner.definitions.is_empty()
    }

    /// Canonically ordered definitions.
    pub fn definitions(&self) -> &[CapabilityDefinitionV1] {
        self.inner.definitions()
    }

    /// Resolve one stable capability identity to its exact definition revision.
    pub fn definition(&self, capability_id: CapabilityId) -> Option<&CapabilityDefinitionV1> {
        self.index
            .get(&capability_id)
            .map(|index| &self.inner.definitions[*index])
    }

    /// Raw serializable snapshot.
    pub fn as_raw(&self) -> &CapabilityGraphSnapshotV1 {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CapabilityGraphError {
    #[error("unsupported capability graph snapshot schema: {0}")]
    UnsupportedSchema(String),
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    #[error("capability graph contains duplicate stable capability identity")]
    DuplicateCapabilityDefinition,
    #[error("capability graph definitions must be in canonical capability-id order")]
    NonCanonicalDefinitionOrder,
    #[error("capability {source:?} references undefined capability {missing:?}")]
    MissingCapabilityReference {
        source: CapabilityId,
        missing: CapabilityId,
    },
    #[error("stored capability graph snapshot identity does not match canonical definitions")]
    SnapshotIdentityMismatch,
}

fn validate_unique_definitions(
    definitions: &[CapabilityDefinitionV1],
) -> Result<(), CapabilityGraphError> {
    if definitions
        .windows(2)
        .any(|pair| pair[0].id() == pair[1].id())
    {
        return Err(CapabilityGraphError::DuplicateCapabilityDefinition);
    }
    Ok(())
}

fn validate_closed_world(
    definitions: &[CapabilityDefinitionV1],
) -> Result<(), CapabilityGraphError> {
    let known: BTreeMap<CapabilityId, CapabilityDefinitionId> = definitions
        .iter()
        .map(|definition| (definition.id(), definition.definition_id()))
        .collect();

    for definition in definitions {
        for referenced in definition.referenced_capabilities() {
            if !known.contains_key(&referenced) {
                return Err(CapabilityGraphError::MissingCapabilityReference {
                    source: definition.id(),
                    missing: referenced,
                });
            }
        }
    }

    Ok(())
}

fn hash_snapshot(definitions: &[CapabilityDefinitionV1]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(64 + definitions.len() * 64);
    put_str(&mut bytes, CAPABILITY_GRAPH_SNAPSHOT_SCHEMA_V1);
    put_str(&mut bytes, CAPABILITY_DEFINITION_SCHEMA_V1);
    put_len(&mut bytes, definitions.len());
    for definition in definitions {
        bytes.extend_from_slice(definition.id().as_bytes());
        bytes.extend_from_slice(definition.definition_id().as_bytes());
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(GRAPH_SNAPSHOT_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilityRequirementV1;

    fn id(name: &str) -> CapabilityId {
        CapabilityId::new("org.example", name).unwrap()
    }

    fn leaf_definition(name: &str) -> CapabilityDefinitionV1 {
        CapabilityDefinitionV1::new("org.example", name, None).unwrap()
    }

    #[test]
    fn constructor_canonicalizes_definition_order() {
        let a = leaf_definition("a");
        let b = leaf_definition("b");

        let left = CapabilityGraphSnapshotV1::new(vec![a.clone(), b.clone()]).unwrap();
        let right = CapabilityGraphSnapshotV1::new(vec![b, a]).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.id(), right.id());
    }

    #[test]
    fn closed_world_rejects_missing_capability_reference() {
        let missing = id("missing");
        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::leaf(missing)),
        )
        .unwrap();

        assert_eq!(
            CapabilityGraphSnapshotV1::new(vec![target.clone()]),
            Err(CapabilityGraphError::MissingCapabilityReference {
                source: target.id(),
                missing,
            })
        );
    }

    #[test]
    fn duplicate_stable_capability_identity_is_rejected_even_for_distinct_revisions() {
        let dependency = id("dependency");
        let v1 = CapabilityDefinitionV1::new("org.example", "target", None).unwrap();
        let v2 = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(CapabilityRequirementV1::leaf(dependency)),
        )
        .unwrap();
        let dependency_definition = leaf_definition("dependency");

        assert_eq!(
            CapabilityGraphSnapshotV1::new(vec![v1, v2, dependency_definition]),
            Err(CapabilityGraphError::DuplicateCapabilityDefinition)
        );
    }

    #[test]
    fn cycles_are_valid_closed_world_structure() {
        let a_id = id("a");
        let b_id = id("b");

        let a = CapabilityDefinitionV1::new(
            "org.example",
            "a",
            Some(CapabilityRequirementV1::leaf(b_id)),
        )
        .unwrap();
        let b = CapabilityDefinitionV1::new(
            "org.example",
            "b",
            Some(CapabilityRequirementV1::leaf(a_id)),
        )
        .unwrap();

        let graph = CapabilityGraphSnapshotV1::new(vec![a, b]).unwrap();
        let validated = graph.validate().unwrap();

        assert_eq!(validated.len(), 2);
        assert!(validated.definition(a_id).is_some());
        assert!(validated.definition(b_id).is_some());
    }

    #[test]
    fn all_of_and_any_of_references_are_both_checked_for_closure() {
        let a = leaf_definition("a");
        let b = leaf_definition("b");
        let c = leaf_definition("c");

        let target = CapabilityDefinitionV1::new(
            "org.example",
            "target",
            Some(
                CapabilityRequirementV1::all_of(vec![
                    CapabilityRequirementV1::leaf(a.id()),
                    CapabilityRequirementV1::any_of(vec![
                        CapabilityRequirementV1::leaf(b.id()),
                        CapabilityRequirementV1::leaf(c.id()),
                    ])
                    .unwrap(),
                ])
                .unwrap(),
            ),
        )
        .unwrap();

        let graph = CapabilityGraphSnapshotV1::new(vec![target, c, a, b]).unwrap();
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn transported_noncanonical_definition_order_is_rejected() {
        let a = leaf_definition("a");
        let b = leaf_definition("b");
        let mut graph = CapabilityGraphSnapshotV1::new(vec![a, b]).unwrap();

        graph.definitions.reverse();

        assert_eq!(
            graph.validate(),
            Err(CapabilityGraphError::NonCanonicalDefinitionOrder)
        );
    }

    #[test]
    fn transported_snapshot_identity_mismatch_is_rejected() {
        let mut graph = CapabilityGraphSnapshotV1::new(vec![leaf_definition("a")]).unwrap();
        graph.snapshot_id = CapabilityGraphSnapshotId([7; 32]);

        assert_eq!(
            graph.validate(),
            Err(CapabilityGraphError::SnapshotIdentityMismatch)
        );
    }

    #[test]
    fn validated_wrapper_resolves_exact_definition() {
        let a = leaf_definition("a");
        let a_id = a.id();
        let a_definition_id = a.definition_id();

        let validated = CapabilityGraphSnapshotV1::new(vec![a])
            .unwrap()
            .validate()
            .unwrap();

        let resolved = validated.definition(a_id).unwrap();
        assert_eq!(resolved.definition_id(), a_definition_id);
    }
}
