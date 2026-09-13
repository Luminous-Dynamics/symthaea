// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical evaluator-owned identity grammar for EUREKA-002 V2 evidence rows.
//!
//! Row identity is derived from canonical public transition provenance. Callers
//! cannot supply arbitrary row-ID bytes. Semantic transition equality is kept
//! separate from row identity so cross-partition duplication remains visible.

use super::hidden_world::PublicAction;
use super::v2_public_schema::{
    V2PublicFamily, V2PublicSchemaError, V2PublicState, action_index, public_schema_commitment,
};

pub(super) const V2_ROW_IDENTITY_REVISION: &str = "EUREKA.002.V2.ROW_IDENTITY.v1";
pub(super) const V2_TRANSITION_SEMANTICS_REVISION: &str =
    "EUREKA.002.V2.TRANSITION_SEMANTICS.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum V2EvidencePartition {
    Development,
    Calibration,
    HeldOut,
    ExternalReplication,
}

impl V2EvidencePartition {
    pub(super) const fn tag(self) -> u8 {
        match self {
            Self::Development => 1,
            Self::Calibration => 2,
            Self::HeldOut => 3,
            Self::ExternalReplication => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2EvidenceIdentityError {
    PublicSchema(V2PublicSchemaError),
    FamilyContextMismatch,
    ContextChangedAcrossTransition,
}

impl From<V2PublicSchemaError> for V2EvidenceIdentityError {
    fn from(value: V2PublicSchemaError) -> Self {
        Self::PublicSchema(value)
    }
}

/// Derive the one canonical row identity from exact public transition
/// provenance. No caller-supplied row identity or redundant mode label exists.
pub(super) fn canonical_row_identity(
    family: V2PublicFamily,
    partition: V2EvidencePartition,
    pre: V2PublicState,
    action: PublicAction,
    post: V2PublicState,
) -> Result<[u8; 32], V2EvidenceIdentityError> {
    validate_transition(family, pre, action, post)?;

    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_ROW_IDENTITY_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.push(family.tag());
    bytes.push(partition.tag());
    bytes.extend_from_slice(&(action_index(action)? as u64).to_le_bytes());
    encode_state(&mut bytes, pre);
    encode_state(&mut bytes, post);
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// Exact public transition semantics, deliberately excluding partition and row
/// identity. This is an equality key, not an authentication token.
pub(super) fn canonical_transition_semantics_bytes(
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
    post: V2PublicState,
) -> Result<Vec<u8>, V2EvidenceIdentityError> {
    validate_transition(family, pre, action, post)?;

    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_TRANSITION_SEMANTICS_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.push(family.tag());
    bytes.extend_from_slice(&(action_index(action)? as u64).to_le_bytes());
    encode_state(&mut bytes, pre);
    encode_state(&mut bytes, post);
    Ok(bytes)
}

fn validate_transition(
    family: V2PublicFamily,
    pre: V2PublicState,
    action: PublicAction,
    post: V2PublicState,
) -> Result<(), V2EvidenceIdentityError> {
    action_index(action)?;
    if !pre.belongs_to(family) || !post.belongs_to(family) {
        return Err(V2EvidenceIdentityError::FamilyContextMismatch);
    }
    if pre.context() != post.context() {
        return Err(V2EvidenceIdentityError::ContextChangedAcrossTransition);
    }
    Ok(())
}

fn encode_state(bytes: &mut Vec<u8>, state: V2PublicState) {
    for value in state.fields() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(fields: [i32; 4]) -> V2PublicState {
        V2PublicState::new(fields).unwrap()
    }

    #[test]
    fn canonical_identity_is_deterministic_and_partition_sensitive() {
        let pre = state([3, 4, 5, 0]);
        let post = state([2, 5, 5, 0]);
        let action = PublicAction::Pulse { slot: 0 };
        let a = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            pre,
            action,
            post,
        )
        .unwrap();
        let b = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            pre,
            action,
            post,
        )
        .unwrap();
        let calibration = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Calibration,
            pre,
            action,
            post,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_ne!(a, [0_u8; 32]);
        assert_ne!(a, calibration);
    }

    #[test]
    fn semantic_equality_key_is_partition_independent() {
        let pre = state([3, 4, 5, 0]);
        let post = state([2, 5, 5, 0]);
        let action = PublicAction::Pulse { slot: 0 };
        let semantics = canonical_transition_semantics_bytes(
            V2PublicFamily::PublicFlowV2,
            pre,
            action,
            post,
        )
        .unwrap();
        let development = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            pre,
            action,
            post,
        )
        .unwrap();
        let calibration = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Calibration,
            pre,
            action,
            post,
        )
        .unwrap();
        assert_ne!(development, calibration);
        assert!(!semantics.is_empty());
    }

    #[test]
    fn changing_public_provenance_changes_identity() {
        let pre = state([3, 4, 5, 0]);
        let post = state([2, 5, 5, 0]);
        let base = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            pre,
            PublicAction::Pulse { slot: 0 },
            post,
        )
        .unwrap();
        let changed_action = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            pre,
            PublicAction::Pulse { slot: 1 },
            post,
        )
        .unwrap();
        let changed_pre = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            state([4, 4, 5, 0]),
            PublicAction::Pulse { slot: 0 },
            post,
        )
        .unwrap();
        let changed_post = canonical_row_identity(
            V2PublicFamily::PublicFlowV2,
            V2EvidencePartition::Development,
            pre,
            PublicAction::Pulse { slot: 0 },
            state([2, 6, 5, 0]),
        )
        .unwrap();
        assert_ne!(base, changed_action);
        assert_ne!(base, changed_pre);
        assert_ne!(base, changed_post);
    }

    #[test]
    fn invalid_context_or_transition_context_change_fails_before_identity() {
        assert_eq!(
            canonical_row_identity(
                V2PublicFamily::PublicFlowV2,
                V2EvidencePartition::Development,
                state([1, 2, 3, 4]),
                PublicAction::NoOp,
                state([1, 2, 3, 4]),
            ),
            Err(V2EvidenceIdentityError::FamilyContextMismatch)
        );
        assert_eq!(
            canonical_row_identity(
                V2PublicFamily::PublicFlowV2,
                V2EvidencePartition::Development,
                state([1, 2, 3, 0]),
                PublicAction::NoOp,
                state([1, 2, 3, 1]),
            ),
            Err(V2EvidenceIdentityError::ContextChangedAcrossTransition)
        );
    }
}
