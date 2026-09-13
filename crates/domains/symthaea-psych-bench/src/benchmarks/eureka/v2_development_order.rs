// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Target-blind sequential learning order for EUREKA-002 V2 Development.
//!
//! The canonical corpus materializer defines which rows belong to Development.
//! This module defines only the deterministic order in which a sequential
//! learner may consume those already-frozen rows and packages the exact static
//! lineage needed by the Development runner. Target state and predictions are
//! never inputs to this plan.

use super::v2_comparator_custody::{
    V2CorpusPartition, V2DevelopmentFitCorpus, V2PublicTransitionEvidence,
};
use super::v2_corpus_schedule::{
    V2ScheduleMaterializationError, V2SchedulePartition, V2ScheduledRow,
    canonical_schedule_root, materialize_all_rows, materialize_family_rows,
};
use super::v2_public_schema::{V2PublicFamily, public_schema_commitment};

pub(super) const V2_DEVELOPMENT_ORDER_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_ORDER.v1";
pub(super) const V2_DEVELOPMENT_ORDER_ROOT_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_ORDER_ROOT.v1";
pub(super) const V2_DEVELOPMENT_PLAN_REVISION: &str =
    "EUREKA.002.V2.DEVELOPMENT_PLAN.v1";
pub(super) const V2_DEVELOPMENT_ROWS_PER_FAMILY: usize = 256;
pub(super) const V2_DEVELOPMENT_ROWS_TOTAL: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2DevelopmentPlan {
    ordered_rows: Vec<V2ScheduledRow>,
    development_corpus: V2DevelopmentFitCorpus,
    full_schedule_root: [u8; 32],
    development_order_root: [u8; 32],
    commitment: [u8; 32],
}

impl V2DevelopmentPlan {
    pub(super) fn ordered_rows(&self) -> &[V2ScheduledRow] {
        &self.ordered_rows
    }

    pub(super) const fn development_corpus(&self) -> &V2DevelopmentFitCorpus {
        &self.development_corpus
    }

    pub(super) const fn full_schedule_root(&self) -> [u8; 32] {
        self.full_schedule_root
    }

    pub(super) const fn development_order_root(&self) -> [u8; 32] {
        self.development_order_root
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

pub(super) fn materialize_development_plan(
) -> Result<V2DevelopmentPlan, V2ScheduleMaterializationError> {
    let ordered_rows = materialize_development_order()?;
    let development_corpus = development_corpus_from_ordered_rows(&ordered_rows)?;
    let full_schedule_root = canonical_schedule_root(&materialize_all_rows()?);
    let development_order_root = development_order_root(&ordered_rows);
    let commitment = development_plan_commitment(
        development_corpus.commitment(),
        full_schedule_root,
        development_order_root,
    );
    Ok(V2DevelopmentPlan {
        ordered_rows,
        development_corpus,
        full_schedule_root,
        development_order_root,
        commitment,
    })
}

pub(super) fn materialize_development_order(
) -> Result<Vec<V2ScheduledRow>, V2ScheduleMaterializationError> {
    let mut flow = development_family_rows(V2PublicFamily::PublicFlowV2)?;
    let mut relay = development_family_rows(V2PublicFamily::PublicRelayV2)?;
    flow.sort_by_key(|row| development_order_key(*row));
    relay.sort_by_key(|row| development_order_key(*row));

    debug_assert_eq!(flow.len(), V2_DEVELOPMENT_ROWS_PER_FAMILY);
    debug_assert_eq!(relay.len(), V2_DEVELOPMENT_ROWS_PER_FAMILY);

    let mut ordered = Vec::with_capacity(V2_DEVELOPMENT_ROWS_TOTAL);
    for (flow_row, relay_row) in flow.into_iter().zip(relay) {
        ordered.push(flow_row);
        ordered.push(relay_row);
    }
    Ok(ordered)
}

fn development_family_rows(
    family: V2PublicFamily,
) -> Result<Vec<V2ScheduledRow>, V2ScheduleMaterializationError> {
    Ok(materialize_family_rows(family)?
        .into_iter()
        .filter(|row| row.partition() == V2SchedulePartition::Development)
        .collect())
}

fn development_corpus_from_ordered_rows(
    rows: &[V2ScheduledRow],
) -> Result<V2DevelopmentFitCorpus, V2ScheduleMaterializationError> {
    let records = rows
        .iter()
        .copied()
        .map(|row| {
            V2PublicTransitionEvidence::new(
                row.family(),
                V2CorpusPartition::Development,
                row.pre(),
                row.action(),
                row.post(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(V2DevelopmentFitCorpus::freeze(records)?)
}

fn development_order_key(row: V2ScheduledRow) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_DEVELOPMENT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&row.row_identity());
    *blake3::hash(&bytes).as_bytes()
}

pub(super) fn development_order_root(rows: &[V2ScheduledRow]) -> [u8; 32] {
    let identities: Vec<_> = rows.iter().map(|row| row.row_identity()).collect();
    development_order_root_from_identities(&identities)
}

fn development_order_root_from_identities(identities: &[[u8; 32]]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_DEVELOPMENT_ORDER_ROOT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, V2_DEVELOPMENT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&(identities.len() as u64).to_le_bytes());
    for identity in identities {
        bytes.extend_from_slice(identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn development_plan_commitment(
    development_corpus_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    development_order_root: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_DEVELOPMENT_PLAN_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&development_corpus_commitment);
    bytes.extend_from_slice(&full_schedule_root);
    bytes.extend_from_slice(&development_order_root);
    bytes.extend_from_slice(&(V2_DEVELOPMENT_ROWS_TOTAL as u64).to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::v2_corpus_schedule::materialize_canonical_corpora;
    use super::super::v2_public_schema::V2_ACTION_COUNT;

    #[test]
    fn canonical_development_order_is_exact_balanced_and_deterministic() {
        let first = materialize_development_order().unwrap();
        let second = materialize_development_order().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), V2_DEVELOPMENT_ROWS_TOTAL);
        assert_eq!(development_order_root(&first), development_order_root(&second));

        for pair in first.chunks_exact(2) {
            assert_eq!(pair[0].family(), V2PublicFamily::PublicFlowV2);
            assert_eq!(pair[1].family(), V2PublicFamily::PublicRelayV2);
            assert_eq!(pair[0].partition(), V2SchedulePartition::Development);
            assert_eq!(pair[1].partition(), V2SchedulePartition::Development);
        }

        assert_eq!(
            first
                .iter()
                .filter(|row| row.family() == V2PublicFamily::PublicFlowV2)
                .count(),
            V2_DEVELOPMENT_ROWS_PER_FAMILY
        );
        assert_eq!(
            first
                .iter()
                .filter(|row| row.family() == V2PublicFamily::PublicRelayV2)
                .count(),
            V2_DEVELOPMENT_ROWS_PER_FAMILY
        );

        let mut action_histogram = vec![0_usize; usize::from(V2_ACTION_COUNT)];
        for row in &first {
            action_histogram[usize::from(row.action_index())] += 1;
        }
        assert_eq!(action_histogram, vec![128, 128, 128, 128]);
    }

    #[test]
    fn development_plan_matches_canonical_materializer_lineage() {
        let plan = materialize_development_plan().unwrap();
        let corpora = materialize_canonical_corpora().unwrap();
        assert_eq!(plan.ordered_rows().len(), V2_DEVELOPMENT_ROWS_TOTAL);
        assert_eq!(
            plan.development_corpus().commitment(),
            corpora.development().commitment()
        );
        assert_eq!(plan.full_schedule_root(), corpora.schedule_root());
        assert_eq!(
            plan.development_order_root(),
            development_order_root(plan.ordered_rows())
        );
        assert_ne!(plan.commitment(), [0_u8; 32]);
    }

    #[test]
    fn ordered_root_binds_sequence_not_just_set_membership() {
        let rows = materialize_development_order().unwrap();
        let canonical = development_order_root(&rows);
        let mut swapped = rows.clone();
        swapped.swap(0, 2);
        assert_ne!(canonical, development_order_root(&swapped));
    }

    #[test]
    fn ordered_root_changes_when_one_identity_changes() {
        let rows = materialize_development_order().unwrap();
        let mut identities: Vec<_> = rows.iter().map(|row| row.row_identity()).collect();
        let original = development_order_root_from_identities(&identities);
        identities[17][9] ^= 0x80;
        assert_ne!(original, development_order_root_from_identities(&identities));
    }
}
