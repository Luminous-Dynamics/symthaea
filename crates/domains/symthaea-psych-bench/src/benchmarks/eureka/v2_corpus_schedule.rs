// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical target-blind EUREKA-002 V2 schedule materialization.
//!
//! This module is the sole downstream corpus authority for the frozen V2
//! construct schedule. It generates public transitions and typed partition
//! corpora only; it cannot train a target or execute comparator/target outcomes.

use std::collections::BTreeMap;

use super::hidden_world::PublicAction;
use super::v2_comparator_custody::{
    V2ComparatorCustodyError, V2CorpusPartition, V2DevelopmentFitCorpus,
    V2PublicTransitionEvidence,
};
use super::v2_evidence_identity::{
    V2EvidenceIdentityError, V2EvidencePartition, canonical_row_identity,
};
use super::v2_public_schema::{
    V2_ACTION_COUNT, V2_COUNT_CARDINALITY, V2_PUBLIC_MODES_PER_FAMILY, V2PublicFamily,
    V2PublicSchemaError, V2PublicState, action_from_index, public_schema_commitment,
};
use super::v2_selection_authorization::{
    V2CalibrationCorpus, V2CalibrationEvidence, V2SelectionAuthorizationError,
};

pub(super) const V2_CANONICAL_SCHEDULE_REVISION: &str =
    "EUREKA.002.V2.CONSTRUCT_SCHEDULE.prototype.v5";
pub(super) const V2_CANONICAL_SCHEDULE_ROOT_REVISION: &str =
    "EUREKA.002.V2.CANONICAL_SCHEDULE_ROOT.v1";
pub(super) const V2_MAX_REALIZED_VALUE_HEADROOM: u16 = 2;
pub(super) const V2_PRE_STATE_CARDINALITY: u16 =
    V2_COUNT_CARDINALITY - V2_MAX_REALIZED_VALUE_HEADROOM;
pub(super) const V2_ROWS_PER_STRATUM: usize = 30;
pub(super) const V2_DEVELOPMENT_PER_STRATUM: usize = 16;
pub(super) const V2_CALIBRATION_PER_STRATUM: usize = 8;
pub(super) const V2_HELDOUT_PER_STRATUM: usize = 4;
pub(super) const V2_EXTERNAL_PER_STRATUM: usize = 2;
pub(super) const V2_STRATA_PER_FAMILY: usize =
    (V2_PUBLIC_MODES_PER_FAMILY as usize) * (V2_ACTION_COUNT as usize);
pub(super) const V2_ROWS_PER_FAMILY: usize =
    V2_STRATA_PER_FAMILY * V2_ROWS_PER_STRATUM;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum V2SchedulePartition {
    Development,
    Calibration,
    HeldOut,
    ExternalReplication,
}

impl V2SchedulePartition {
    pub(super) const ALL: [Self; 4] = [
        Self::Development,
        Self::Calibration,
        Self::HeldOut,
        Self::ExternalReplication,
    ];

    fn range(self) -> std::ops::Range<usize> {
        match self {
            Self::Development => 0..V2_DEVELOPMENT_PER_STRATUM,
            Self::Calibration => {
                V2_DEVELOPMENT_PER_STRATUM
                    ..V2_DEVELOPMENT_PER_STRATUM + V2_CALIBRATION_PER_STRATUM
            }
            Self::HeldOut => {
                V2_DEVELOPMENT_PER_STRATUM + V2_CALIBRATION_PER_STRATUM
                    ..V2_DEVELOPMENT_PER_STRATUM
                        + V2_CALIBRATION_PER_STRATUM
                        + V2_HELDOUT_PER_STRATUM
            }
            Self::ExternalReplication => {
                V2_DEVELOPMENT_PER_STRATUM
                    + V2_CALIBRATION_PER_STRATUM
                    + V2_HELDOUT_PER_STRATUM
                    ..V2_ROWS_PER_STRATUM
            }
        }
    }

    pub(super) const fn expected_per_stratum(self) -> usize {
        match self {
            Self::Development => V2_DEVELOPMENT_PER_STRATUM,
            Self::Calibration => V2_CALIBRATION_PER_STRATUM,
            Self::HeldOut => V2_HELDOUT_PER_STRATUM,
            Self::ExternalReplication => V2_EXTERNAL_PER_STRATUM,
        }
    }

    pub(super) const fn evidence_partition(self) -> V2EvidencePartition {
        match self {
            Self::Development => V2EvidencePartition::Development,
            Self::Calibration => V2EvidencePartition::Calibration,
            Self::HeldOut => V2EvidencePartition::HeldOut,
            Self::ExternalReplication => V2EvidencePartition::ExternalReplication,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2ScheduledRow {
    family: V2PublicFamily,
    partition: V2SchedulePartition,
    mode: u8,
    action_index: u8,
    action: PublicAction,
    pre: V2PublicState,
    post: V2PublicState,
    row_identity: [u8; 32],
}

impl V2ScheduledRow {
    pub(super) const fn family(self) -> V2PublicFamily {
        self.family
    }

    pub(super) const fn partition(self) -> V2SchedulePartition {
        self.partition
    }

    pub(super) const fn mode(self) -> u8 {
        self.mode
    }

    pub(super) const fn action_index(self) -> u8 {
        self.action_index
    }

    pub(super) const fn action(self) -> PublicAction {
        self.action
    }

    pub(super) const fn pre(self) -> V2PublicState {
        self.pre
    }

    pub(super) const fn post(self) -> V2PublicState {
        self.post
    }

    pub(super) const fn row_identity(self) -> [u8; 32] {
        self.row_identity
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2CanonicalCorpora {
    development: V2DevelopmentFitCorpus,
    calibration: V2CalibrationCorpus,
    schedule_root: [u8; 32],
}

impl V2CanonicalCorpora {
    pub(super) const fn development(&self) -> &V2DevelopmentFitCorpus {
        &self.development
    }

    pub(super) const fn calibration(&self) -> &V2CalibrationCorpus {
        &self.calibration
    }

    pub(super) const fn schedule_root(&self) -> [u8; 32] {
        self.schedule_root
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ScheduleMaterializationError {
    PublicSchema(V2PublicSchemaError),
    EvidenceIdentity(V2EvidenceIdentityError),
    Custody(V2ComparatorCustodyError),
    Selection(V2SelectionAuthorizationError),
    StateGenerationBudget,
    WrongCardinality,
    CrossPartitionInputOverlap,
    CrossPartitionTransitionOverlap,
}

impl From<V2PublicSchemaError> for V2ScheduleMaterializationError {
    fn from(value: V2PublicSchemaError) -> Self {
        Self::PublicSchema(value)
    }
}

impl From<V2EvidenceIdentityError> for V2ScheduleMaterializationError {
    fn from(value: V2EvidenceIdentityError) -> Self {
        Self::EvidenceIdentity(value)
    }
}

impl From<V2ComparatorCustodyError> for V2ScheduleMaterializationError {
    fn from(value: V2ComparatorCustodyError) -> Self {
        Self::Custody(value)
    }
}

impl From<V2SelectionAuthorizationError> for V2ScheduleMaterializationError {
    fn from(value: V2SelectionAuthorizationError) -> Self {
        Self::Selection(value)
    }
}

pub(super) fn materialize_family_rows(
    family: V2PublicFamily,
) -> Result<Vec<V2ScheduledRow>, V2ScheduleMaterializationError> {
    let mut rows = Vec::with_capacity(V2_ROWS_PER_FAMILY);
    for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
        let context = family.context(mode)?;
        for action_index in 0..V2_ACTION_COUNT {
            let action = action_from_index(usize::from(action_index))?;
            let states = stratum_states(family, mode, action_index)?;
            for partition in V2SchedulePartition::ALL {
                for index in partition.range() {
                    let triple = states[index];
                    let pre = V2PublicState::new([triple[0], triple[1], triple[2], context])?;
                    let post = transition(family, pre, action_index)?;
                    let row_identity = canonical_row_identity(
                        family,
                        partition.evidence_partition(),
                        pre,
                        action,
                        post,
                    )?;
                    rows.push(V2ScheduledRow {
                        family,
                        partition,
                        mode,
                        action_index,
                        action,
                        pre,
                        post,
                        row_identity,
                    });
                }
            }
        }
    }
    if rows.len() != V2_ROWS_PER_FAMILY {
        return Err(V2ScheduleMaterializationError::WrongCardinality);
    }
    validate_partition_novelty(&rows)?;
    Ok(rows)
}

pub(super) fn materialize_all_rows(
) -> Result<Vec<V2ScheduledRow>, V2ScheduleMaterializationError> {
    let mut rows = Vec::with_capacity(V2_ROWS_PER_FAMILY * V2PublicFamily::ALL.len());
    for family in V2PublicFamily::ALL {
        rows.extend(materialize_family_rows(family)?);
    }
    Ok(rows)
}

pub(super) fn materialize_canonical_corpora(
) -> Result<V2CanonicalCorpora, V2ScheduleMaterializationError> {
    let rows = materialize_all_rows()?;
    let development = development_corpus_from_rows(&rows)?;
    let calibration = calibration_corpus_from_rows(&development, &rows)?;
    let schedule_root = canonical_schedule_root(&rows);
    Ok(V2CanonicalCorpora {
        development,
        calibration,
        schedule_root,
    })
}

fn development_corpus_from_rows(
    rows: &[V2ScheduledRow],
) -> Result<V2DevelopmentFitCorpus, V2ScheduleMaterializationError> {
    let records = rows
        .iter()
        .copied()
        .filter(|row| row.partition == V2SchedulePartition::Development)
        .map(|row| {
            V2PublicTransitionEvidence::new(
                row.family,
                V2CorpusPartition::Development,
                row.pre,
                row.action,
                row.post,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(V2DevelopmentFitCorpus::freeze(records)?)
}

fn calibration_corpus_from_rows(
    development: &V2DevelopmentFitCorpus,
    rows: &[V2ScheduledRow],
) -> Result<V2CalibrationCorpus, V2ScheduleMaterializationError> {
    let records = rows
        .iter()
        .copied()
        .filter(|row| row.partition == V2SchedulePartition::Calibration)
        .map(|row| V2CalibrationEvidence::new(row.family, row.pre, row.action, row.post))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(V2CalibrationCorpus::freeze(development, records)?)
}

pub(super) fn canonical_schedule_root(rows: &[V2ScheduledRow]) -> [u8; 32] {
    let mut identities: Vec<_> = rows.iter().map(|row| row.row_identity).collect();
    identities.sort_unstable();
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CANONICAL_SCHEDULE_ROOT_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&(identities.len() as u64).to_le_bytes());
    for identity in identities {
        bytes.extend_from_slice(&identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn stratum_states(
    family: V2PublicFamily,
    mode: u8,
    action_index: u8,
) -> Result<Vec<[i32; 3]>, V2ScheduleMaterializationError> {
    let mut states = Vec::with_capacity(V2_ROWS_PER_STRATUM);
    let mut seen = std::collections::BTreeSet::new();
    for ordinal in 0..4096_u32 {
        let digest = schedule_digest(b"public-state", family, mode, action_index, ordinal);
        let bytes = digest.as_bytes();
        let state = [
            i32::from(u16::from_le_bytes([bytes[0], bytes[1]]) % V2_PRE_STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[2], bytes[3]]) % V2_PRE_STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[4], bytes[5]]) % V2_PRE_STATE_CARDINALITY),
        ];
        if seen.insert(state) {
            states.push(state);
            if states.len() == V2_ROWS_PER_STRATUM {
                return Ok(states);
            }
        }
    }
    Err(V2ScheduleMaterializationError::StateGenerationBudget)
}

fn schedule_digest(
    purpose: &[u8],
    family: V2PublicFamily,
    mode: u8,
    action_index: u8,
    ordinal: u32,
) -> blake3::Hash {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_CANONICAL_SCHEDULE_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    encode_bytes(&mut bytes, purpose);
    bytes.push(family.tag());
    bytes.push(mode);
    bytes.push(action_index);
    bytes.extend_from_slice(&ordinal.to_le_bytes());
    blake3::hash(&bytes)
}

fn transition(
    family: V2PublicFamily,
    pre: V2PublicState,
    action_index: u8,
) -> Result<V2PublicState, V2ScheduleMaterializationError> {
    match family {
        V2PublicFamily::PublicFlowV2 => {
            let [x, y, z, mode] = pre.fields();
            let mut values = [x, y, z];
            match action_index {
                0 => {}
                1 => transfer_one(&mut values, 0, 1),
                2 => transfer_one(&mut values, 1, 2),
                3 => transfer_one(&mut values, 2, 0),
                _ => return Err(V2ScheduleMaterializationError::WrongCardinality),
            }
            match mode {
                0 => {}
                1 => values[0] = values[0].saturating_add(1),
                2 => transfer_one(&mut values, 0, 1),
                3 => transfer_one(&mut values, 1, 2),
                _ => return Err(V2ScheduleMaterializationError::WrongCardinality),
            }
            Ok(V2PublicState::new([values[0], values[1], values[2], mode])?)
        }
        V2PublicFamily::PublicRelayV2 => {
            let [mut x, mut y, mut z, rule] = pre.fields();
            match action_index {
                0 => {}
                1 => x = x.saturating_add(1),
                2 => y = y.saturating_add(1),
                3 => z = z.saturating_add(1),
                _ => return Err(V2ScheduleMaterializationError::WrongCardinality),
            }
            match rule {
                4 => {}
                5 => y = x,
                6 => z = y,
                7 => x = z,
                _ => return Err(V2ScheduleMaterializationError::WrongCardinality),
            }
            Ok(V2PublicState::new([x, y, z, rule])?)
        }
    }
}

fn transfer_one(values: &mut [i32; 3], from: usize, to: usize) {
    if values[from] > 0 {
        values[from] -= 1;
        values[to] += 1;
    }
}

fn validate_partition_novelty(
    rows: &[V2ScheduledRow],
) -> Result<(), V2ScheduleMaterializationError> {
    let mut input_partitions = BTreeMap::<(u8, [i32; 4], u8), V2SchedulePartition>::new();
    let mut transition_partitions =
        BTreeMap::<(u8, [i32; 4], u8, [i32; 4]), V2SchedulePartition>::new();
    for row in rows {
        let input_key = (row.family.tag(), row.pre.fields(), row.action_index);
        if let Some(previous) = input_partitions.insert(input_key, row.partition) {
            if previous != row.partition {
                return Err(V2ScheduleMaterializationError::CrossPartitionInputOverlap);
            }
        }
        let transition_key = (
            row.family.tag(),
            row.pre.fields(),
            row.action_index,
            row.post.fields(),
        );
        if let Some(previous) = transition_partitions.insert(transition_key, row.partition) {
            if previous != row.partition {
                return Err(V2ScheduleMaterializationError::CrossPartitionTransitionOverlap);
            }
        }
    }
    Ok(())
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_partition_cardinalities_materialize() {
        let rows = materialize_all_rows().unwrap();
        assert_eq!(rows.len(), 960);
        for family in V2PublicFamily::ALL {
            let family_rows: Vec<_> = rows
                .iter()
                .filter(|row| row.family == family)
                .collect();
            assert_eq!(family_rows.len(), V2_ROWS_PER_FAMILY);
            for partition in V2SchedulePartition::ALL {
                assert_eq!(
                    family_rows
                        .iter()
                        .filter(|row| row.partition == partition)
                        .count(),
                    partition.expected_per_stratum() * V2_STRATA_PER_FAMILY
                );
            }
        }
        assert_eq!(
            rows.iter()
                .filter(|row| row.partition == V2SchedulePartition::Development)
                .count(),
            512
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.partition == V2SchedulePartition::Calibration)
                .count(),
            256
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.partition == V2SchedulePartition::HeldOut)
                .count(),
            128
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.partition == V2SchedulePartition::ExternalReplication)
                .count(),
            64
        );
    }

    #[test]
    fn materialized_actual_states_and_identities_are_canonical() {
        for row in materialize_all_rows().unwrap() {
            assert!(V2PublicState::new(row.pre.fields()).is_ok());
            assert!(V2PublicState::new(row.post.fields()).is_ok());
            assert_eq!(
                row.row_identity,
                canonical_row_identity(
                    row.family,
                    row.partition.evidence_partition(),
                    row.pre,
                    row.action,
                    row.post,
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn canonical_typed_corpora_are_deterministic_and_exact_size() {
        let a = materialize_canonical_corpora().unwrap();
        let b = materialize_canonical_corpora().unwrap();
        assert_eq!(a, b);
        assert_eq!(a.development.records().len(), 512);
        assert_eq!(a.calibration.len(), 256);
        assert_ne!(a.schedule_root(), [0_u8; 32]);
        assert_eq!(a.development.commitment(), b.development.commitment());
        assert_eq!(a.calibration.commitment(), b.calibration.commitment());
        assert_eq!(a.calibration.fit_corpus_commitment(), a.development.commitment());
    }

    #[test]
    fn canonical_corpus_commitments_ignore_source_iteration_order() {
        let mut rows = materialize_all_rows().unwrap();
        let development = development_corpus_from_rows(&rows).unwrap();
        let calibration = calibration_corpus_from_rows(&development, &rows).unwrap();
        rows.reverse();
        let development_reversed = development_corpus_from_rows(&rows).unwrap();
        let calibration_reversed =
            calibration_corpus_from_rows(&development_reversed, &rows).unwrap();
        assert_eq!(development.commitment(), development_reversed.commitment());
        assert_eq!(calibration.commitment(), calibration_reversed.commitment());
    }

    #[test]
    fn generated_headroom_keeps_all_realized_values_inside_public_domain() {
        assert_eq!(V2_MAX_REALIZED_VALUE_HEADROOM, 2);
        assert_eq!(V2_PRE_STATE_CARDINALITY, V2_COUNT_CARDINALITY - 2);
        for row in materialize_all_rows().unwrap() {
            assert!(row.pre.fields()[..3].iter().all(|value| *value <= 29));
            assert!(row.post.fields()[..3].iter().all(|value| *value <= 31));
        }
    }
}
