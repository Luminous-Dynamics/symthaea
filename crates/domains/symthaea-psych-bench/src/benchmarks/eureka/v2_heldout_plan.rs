// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target-blind HeldOut plan for EUREKA-002 V2.
//!
//! This module freezes which already-materialized HeldOut rows may be used and
//! their deterministic family-balanced order. It owns no target/comparator
//! prediction authority and performs no evaluation.

use super::v2_corpus_schedule::{
    V2ScheduleMaterializationError, V2SchedulePartition, V2ScheduledRow,
    canonical_schedule_root, materialize_all_rows, materialize_family_rows,
};
use super::v2_public_schema::{V2_ACTION_COUNT, V2PublicFamily, public_schema_commitment};

pub(super) const V2_HELDOUT_ORDER_REVISION: &str = "EUREKA.002.V2.HELDOUT_ORDER.v1";
pub(super) const V2_HELDOUT_ROOT_REVISION: &str = "EUREKA.002.V2.HELDOUT_ROOT.v1";
pub(super) const V2_HELDOUT_PLAN_REVISION: &str = "EUREKA.002.V2.HELDOUT_PLAN.v1";
pub(super) const V2_HELDOUT_ROWS_PER_FAMILY: usize = 64;
pub(super) const V2_HELDOUT_ROWS_TOTAL: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2HeldOutPlan {
    ordered_rows: Vec<V2ScheduledRow>,
    full_schedule_root: [u8; 32],
    ordered_root: [u8; 32],
    commitment: [u8; 32],
}

impl V2HeldOutPlan {
    pub(super) fn ordered_rows(&self) -> &[V2ScheduledRow] {
        &self.ordered_rows
    }

    pub(super) const fn full_schedule_root(&self) -> [u8; 32] {
        self.full_schedule_root
    }

    pub(super) const fn ordered_root(&self) -> [u8; 32] {
        self.ordered_root
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

pub(super) fn materialize_heldout_plan(
) -> Result<V2HeldOutPlan, V2ScheduleMaterializationError> {
    let mut rows = Vec::with_capacity(V2_HELDOUT_ROWS_TOTAL);
    for family in V2PublicFamily::ALL {
        rows.extend(
            materialize_family_rows(family)?
                .into_iter()
                .filter(|row| row.partition() == V2SchedulePartition::HeldOut),
        );
    }
    let full_schedule_root = canonical_schedule_root(&materialize_all_rows()?);
    Ok(freeze_heldout_rows(rows, full_schedule_root))
}

fn freeze_heldout_rows(
    rows: Vec<V2ScheduledRow>,
    full_schedule_root: [u8; 32],
) -> V2HeldOutPlan {
    let mut flow: Vec<_> = rows
        .iter()
        .copied()
        .filter(|row| row.family() == V2PublicFamily::PublicFlowV2)
        .collect();
    let mut relay: Vec<_> = rows
        .iter()
        .copied()
        .filter(|row| row.family() == V2PublicFamily::PublicRelayV2)
        .collect();
    flow.sort_by_key(|row| heldout_order_key(*row));
    relay.sort_by_key(|row| heldout_order_key(*row));

    let mut ordered_rows = Vec::with_capacity(flow.len().saturating_add(relay.len()));
    for (flow_row, relay_row) in flow.into_iter().zip(relay) {
        ordered_rows.push(flow_row);
        ordered_rows.push(relay_row);
    }
    let ordered_root = heldout_root(&ordered_rows);
    let commitment = heldout_plan_commitment(full_schedule_root, ordered_root, ordered_rows.len());
    V2HeldOutPlan {
        ordered_rows,
        full_schedule_root,
        ordered_root,
        commitment,
    }
}

fn heldout_order_key(row: V2ScheduledRow) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_HELDOUT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&row.row_identity());
    *blake3::hash(&bytes).as_bytes()
}

fn heldout_root(rows: &[V2ScheduledRow]) -> [u8; 32] {
    let identities: Vec<_> = rows.iter().map(|row| row.row_identity()).collect();
    heldout_root_from_identities(&identities)
}

fn heldout_root_from_identities(identities: &[[u8; 32]]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_HELDOUT_ROOT_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_HELDOUT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&(identities.len() as u64).to_le_bytes());
    for identity in identities {
        bytes.extend_from_slice(identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn heldout_plan_commitment(
    full_schedule_root: [u8; 32],
    ordered_root: [u8; 32],
    row_count: usize,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_HELDOUT_PLAN_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&full_schedule_root);
    bytes.extend_from_slice(&ordered_root);
    bytes.extend_from_slice(&(row_count as u64).to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_heldout_plan_is_exact_balanced_and_deterministic() {
        let first = materialize_heldout_plan().unwrap();
        let second = materialize_heldout_plan().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.ordered_rows().len(), V2_HELDOUT_ROWS_TOTAL);
        assert_eq!(first.ordered_root(), second.ordered_root());
        assert_ne!(first.commitment(), [0_u8; 32]);

        for pair in first.ordered_rows().chunks_exact(2) {
            assert_eq!(pair[0].family(), V2PublicFamily::PublicFlowV2);
            assert_eq!(pair[1].family(), V2PublicFamily::PublicRelayV2);
            assert_eq!(pair[0].partition(), V2SchedulePartition::HeldOut);
            assert_eq!(pair[1].partition(), V2SchedulePartition::HeldOut);
        }

        let mut family_histogram = [0_usize; 2];
        let mut action_histogram = [0_usize; V2_ACTION_COUNT as usize];
        for row in first.ordered_rows().iter().copied() {
            let family = match row.family() {
                V2PublicFamily::PublicFlowV2 => 0,
                V2PublicFamily::PublicRelayV2 => 1,
            };
            family_histogram[family] += 1;
            action_histogram[usize::from(row.action_index())] += 1;
        }
        assert_eq!(family_histogram, [V2_HELDOUT_ROWS_PER_FAMILY; 2]);
        assert_eq!(action_histogram, [32, 32, 32, 32]);
    }

    #[test]
    fn input_iteration_order_does_not_change_canonical_heldout_plan() {
        let canonical = materialize_heldout_plan().unwrap();
        let mut reversed = canonical.ordered_rows().to_vec();
        reversed.reverse();
        let rebuilt = freeze_heldout_rows(reversed, canonical.full_schedule_root());
        assert_eq!(canonical.ordered_rows(), rebuilt.ordered_rows());
        assert_eq!(canonical.ordered_root(), rebuilt.ordered_root());
        assert_eq!(canonical.commitment(), rebuilt.commitment());
    }

    #[test]
    fn heldout_root_binds_sequence_and_every_identity() {
        let plan = materialize_heldout_plan().unwrap();
        let mut identities: Vec<_> = plan
            .ordered_rows()
            .iter()
            .map(|row| row.row_identity())
            .collect();
        let canonical = heldout_root_from_identities(&identities);
        identities.swap(0, 2);
        assert_ne!(canonical, heldout_root_from_identities(&identities));
        identities.swap(0, 2);
        identities[31][4] ^= 0x40;
        assert_ne!(canonical, heldout_root_from_identities(&identities));
    }
}
