// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent target-blind audit for the EUREKA-002 V2 schedule design.
//!
//! This deliberately does not call the primary `v2_construct` generator or its
//! validators. It independently reproduces only the schedule construction and
//! public dynamics while consuming the one canonical public schema. The
//! construct capsule is therefore not solely responsible for validating itself,
//! but the audit also cannot drift into a second family/action/range grammar.

use std::collections::{BTreeMap, BTreeSet};

use super::v2_public_schema::{
    V2_ACTION_COUNT, V2_COUNT_CARDINALITY, V2_OBSERVATION_DIM, V2_PUBLIC_MODES_PER_FAMILY,
    V2PublicFamily, V2PublicState, public_schema_commitment,
};

const SCHEDULE_REVISION: &str = "EUREKA.002.V2.CONSTRUCT_SCHEDULE.prototype.v5";
const MAX_POSITIVE_FIELD_DELTA: u16 = 2;
const PRE_STATE_CARDINALITY: u16 = V2_COUNT_CARDINALITY - MAX_POSITIVE_FIELD_DELTA;
const PER_STRATUM: usize = 30;
const DEV_PER_STRATUM: usize = 16;
const CAL_PER_STRATUM: usize = 8;
const HELD_PER_STRATUM: usize = 4;
const EXT_PER_STRATUM: usize = 2;
const STRATA: usize =
    (V2_PUBLIC_MODES_PER_FAMILY as usize) * (V2_ACTION_COUNT as usize);
const ROWS_PER_FAMILY: usize = STRATA * PER_STRATUM;
const HISTOGRAM_BINS: usize = 8;
const MAX_HISTOGRAM_TV_BPS: u16 = 5_500;
const MIN_OCCUPIED_BINS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AuditPartition {
    Development,
    Calibration,
    HeldOut,
    External,
}

impl AuditPartition {
    const ALL: [Self; 4] = [
        Self::Development,
        Self::Calibration,
        Self::HeldOut,
        Self::External,
    ];

    fn range(self) -> std::ops::Range<usize> {
        match self {
            Self::Development => 0..DEV_PER_STRATUM,
            Self::Calibration => DEV_PER_STRATUM..DEV_PER_STRATUM + CAL_PER_STRATUM,
            Self::HeldOut => DEV_PER_STRATUM + CAL_PER_STRATUM
                ..DEV_PER_STRATUM + CAL_PER_STRATUM + HELD_PER_STRATUM,
            Self::External => DEV_PER_STRATUM + CAL_PER_STRATUM + HELD_PER_STRATUM..PER_STRATUM,
        }
    }

    const fn expected_per_stratum(self) -> usize {
        match self {
            Self::Development => DEV_PER_STRATUM,
            Self::Calibration => CAL_PER_STRATUM,
            Self::HeldOut => HELD_PER_STRATUM,
            Self::External => EXT_PER_STRATUM,
        }
    }

    const fn expected_rows_per_family(self) -> usize {
        self.expected_per_stratum() * STRATA
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AuditState([i32; V2_OBSERVATION_DIM]);

impl AuditState {
    fn public(fields: [i32; V2_OBSERVATION_DIM]) -> Result<Self, AuditError> {
        V2PublicState::new(fields)
            .map(|_| Self(fields))
            .map_err(|_| AuditError::PublicSchema)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AuditRow {
    family: V2PublicFamily,
    partition: AuditPartition,
    mode: u8,
    action: u8,
    pre: AuditState,
    post: AuditState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuditError {
    PublicSchema,
    StateGenerationBudget,
    WrongCardinality,
    StratumImbalance,
    PublicSufficiency,
    CrossPartitionInputOverlap,
    CrossPartitionTransitionOverlap,
    HistogramDistributionDrift,
    SparsePartitionHistogram,
}

fn transfer_one(values: &mut [i32; 3], from: usize, to: usize) {
    if values[from] > 0 {
        values[from] -= 1;
        values[to] += 1;
    }
}

fn transition(
    family: V2PublicFamily,
    pre: AuditState,
    action: u8,
) -> Result<AuditState, AuditError> {
    match family {
        V2PublicFamily::PublicFlowV2 => {
            let [x, y, z, mode] = pre.0;
            let mut values = [x, y, z];
            match action {
                0 => {}
                1 => transfer_one(&mut values, 0, 1),
                2 => transfer_one(&mut values, 1, 2),
                3 => transfer_one(&mut values, 2, 0),
                _ => return Err(AuditError::PublicSchema),
            }
            match mode {
                0 => {}
                1 => values[0] = values[0].saturating_add(1),
                2 => transfer_one(&mut values, 0, 1),
                3 => transfer_one(&mut values, 1, 2),
                _ => return Err(AuditError::PublicSchema),
            }
            AuditState::public([values[0], values[1], values[2], mode])
        }
        V2PublicFamily::PublicRelayV2 => {
            let [mut x, mut y, mut z, rule] = pre.0;
            match action {
                0 => {}
                1 => x = x.saturating_add(1),
                2 => y = y.saturating_add(1),
                3 => z = z.saturating_add(1),
                _ => return Err(AuditError::PublicSchema),
            }
            match rule {
                4 => {}
                5 => y = x,
                6 => z = y,
                7 => x = z,
                _ => return Err(AuditError::PublicSchema),
            }
            AuditState::public([x, y, z, rule])
        }
    }
}

fn mixed_state_digest(
    family: V2PublicFamily,
    mode: u8,
    action: u8,
    ordinal: u32,
) -> blake3::Hash {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, SCHEDULE_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    encode_bytes(&mut bytes, b"public-state");
    bytes.push(family.tag());
    bytes.push(mode);
    bytes.push(action);
    bytes.extend_from_slice(&ordinal.to_le_bytes());
    blake3::hash(&bytes)
}

fn independent_states(
    family: V2PublicFamily,
    mode: u8,
    action: u8,
) -> Result<Vec<[i32; 3]>, AuditError> {
    let mut states = Vec::with_capacity(PER_STRATUM);
    let mut seen = BTreeSet::new();
    for ordinal in 0..4096_u32 {
        let digest = mixed_state_digest(family, mode, action, ordinal);
        let bytes = digest.as_bytes();
        let triple = [
            i32::from(u16::from_le_bytes([bytes[0], bytes[1]]) % PRE_STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[2], bytes[3]]) % PRE_STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[4], bytes[5]]) % PRE_STATE_CARDINALITY),
        ];
        if seen.insert(triple) {
            states.push(triple);
            if states.len() == PER_STRATUM {
                return Ok(states);
            }
        }
    }
    Err(AuditError::StateGenerationBudget)
}

fn independent_schedule(family: V2PublicFamily) -> Result<Vec<AuditRow>, AuditError> {
    let mut rows = Vec::with_capacity(ROWS_PER_FAMILY);
    for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
        let context = family.context(mode).map_err(|_| AuditError::PublicSchema)?;
        for action in 0..V2_ACTION_COUNT {
            let states = independent_states(family, mode, action)?;
            for partition in AuditPartition::ALL {
                for index in partition.range() {
                    let triple = states[index];
                    let pre = AuditState::public([triple[0], triple[1], triple[2], context])?;
                    let post = transition(family, pre, action)?;
                    rows.push(AuditRow {
                        family,
                        partition,
                        mode,
                        action,
                        pre,
                        post,
                    });
                }
            }
        }
    }
    if rows.len() != ROWS_PER_FAMILY {
        return Err(AuditError::WrongCardinality);
    }
    Ok(rows)
}

fn validate_structure(rows: &[AuditRow]) -> Result<(), AuditError> {
    if rows.len() != ROWS_PER_FAMILY {
        return Err(AuditError::WrongCardinality);
    }
    for partition in AuditPartition::ALL {
        let partition_rows = rows.iter().filter(|row| row.partition == partition).count();
        if partition_rows != partition.expected_rows_per_family() {
            return Err(AuditError::WrongCardinality);
        }
        let expected_per_stratum = partition.expected_per_stratum();
        for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
            for action in 0..V2_ACTION_COUNT {
                let count = rows
                    .iter()
                    .filter(|row| {
                        row.partition == partition && row.mode == mode && row.action == action
                    })
                    .count();
                if count != expected_per_stratum {
                    return Err(AuditError::StratumImbalance);
                }
            }
        }
    }
    Ok(())
}

fn validate_sufficiency_and_novelty(rows: &[AuditRow]) -> Result<(), AuditError> {
    let mut outcome_by_key = BTreeMap::<(V2PublicFamily, AuditState, u8), AuditState>::new();
    let mut partition_by_key =
        BTreeMap::<(V2PublicFamily, AuditState, u8), AuditPartition>::new();
    let mut partition_by_transition =
        BTreeMap::<(V2PublicFamily, AuditState, u8, AuditState), AuditPartition>::new();

    for row in rows {
        let key = (row.family, row.pre, row.action);
        if let Some(previous) = outcome_by_key.insert(key, row.post) {
            if previous != row.post {
                return Err(AuditError::PublicSufficiency);
            }
        }
        if let Some(previous) = partition_by_key.insert(key, row.partition) {
            if previous != row.partition {
                return Err(AuditError::CrossPartitionInputOverlap);
            }
        }
        let transition = (row.family, row.pre, row.action, row.post);
        if let Some(previous) = partition_by_transition.insert(transition, row.partition) {
            if previous != row.partition {
                return Err(AuditError::CrossPartitionTransitionOverlap);
            }
        }
    }
    Ok(())
}

fn histogram(rows: &[AuditRow], partition: AuditPartition, field: usize) -> [u32; HISTOGRAM_BINS] {
    let mut bins = [0_u32; HISTOGRAM_BINS];
    for row in rows.iter().filter(|row| row.partition == partition) {
        let value = row.pre.0[field].clamp(0, i32::from(PRE_STATE_CARDINALITY - 1));
        let bin = usize::try_from(value).expect("nonnegative V2 state") * HISTOGRAM_BINS
            / usize::from(PRE_STATE_CARDINALITY);
        bins[bin.min(HISTOGRAM_BINS - 1)] += 1;
    }
    bins
}

fn histogram_tv_bps(left: &[u32; HISTOGRAM_BINS], right: &[u32; HISTOGRAM_BINS]) -> u16 {
    let left_total: u64 = left.iter().map(|value| u64::from(*value)).sum();
    let right_total: u64 = right.iter().map(|value| u64::from(*value)).sum();
    if left_total == 0 || right_total == 0 {
        return 10_000;
    }
    // TV = 1/2 * sum |p_i - q_i|. Compute exactly in the common
    // denominator left_total * right_total before rounding to basis points.
    let numerator: u128 = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| {
            let lhs = u128::from(*l) * u128::from(right_total);
            let rhs = u128::from(*r) * u128::from(left_total);
            lhs.abs_diff(rhs)
        })
        .sum();
    let denominator = 2_u128 * u128::from(left_total) * u128::from(right_total);
    u16::try_from((numerator * 10_000_u128) / denominator)
        .expect("TV basis points fit u16")
}

fn validate_distribution(rows: &[AuditRow]) -> Result<(), AuditError> {
    for field in 0..3_usize {
        let mut per_partition = Vec::new();
        for partition in AuditPartition::ALL {
            let bins = histogram(rows, partition, field);
            let occupied = bins.iter().filter(|count| **count > 0).count();
            if occupied < MIN_OCCUPIED_BINS {
                return Err(AuditError::SparsePartitionHistogram);
            }
            per_partition.push((partition, bins));
        }
        for left in 0..per_partition.len() {
            for right in left + 1..per_partition.len() {
                let tv = histogram_tv_bps(&per_partition[left].1, &per_partition[right].1);
                if tv > MAX_HISTOGRAM_TV_BPS {
                    return Err(AuditError::HistogramDistributionDrift);
                }
            }
        }
    }
    Ok(())
}

fn validate_actual_public_domain(rows: &[AuditRow]) -> Result<(), AuditError> {
    for row in rows {
        V2PublicState::new(row.pre.0).map_err(|_| AuditError::PublicSchema)?;
        V2PublicState::new(row.post.0).map_err(|_| AuditError::PublicSchema)?;
    }
    Ok(())
}

fn audit(rows: &[AuditRow]) -> Result<(), AuditError> {
    validate_structure(rows)?;
    validate_sufficiency_and_novelty(rows)?;
    validate_distribution(rows)?;
    validate_actual_public_domain(rows)?;
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
    fn independent_generator_passes_structure_novelty_histogram_and_domain_gates() {
        for family in V2PublicFamily::ALL {
            let rows = independent_schedule(family).unwrap();
            audit(&rows).unwrap();
        }
    }

    #[test]
    fn independent_generator_enforces_the_two_count_transition_margin() {
        assert_eq!(MAX_POSITIVE_FIELD_DELTA, 2);
        assert_eq!(PRE_STATE_CARDINALITY, V2_COUNT_CARDINALITY - 2);
        for family in V2PublicFamily::ALL {
            let rows = independent_schedule(family).unwrap();
            assert!(rows.iter().all(|row| V2PublicState::new(row.pre.0).is_ok()));
            assert!(rows.iter().all(|row| V2PublicState::new(row.post.0).is_ok()));
        }
    }

    #[test]
    fn partition_is_not_an_input_to_public_state_generation() {
        for family in V2PublicFamily::ALL {
            for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
                for action in 0..V2_ACTION_COUNT {
                    let states = independent_states(family, mode, action).unwrap();
                    assert_eq!(states.len(), PER_STRATUM);
                    let dev = &states[AuditPartition::Development.range()];
                    let cal = &states[AuditPartition::Calibration.range()];
                    let held = &states[AuditPartition::HeldOut.range()];
                    let ext = &states[AuditPartition::External.range()];
                    assert_eq!(dev.len(), DEV_PER_STRATUM);
                    assert_eq!(cal.len(), CAL_PER_STRATUM);
                    assert_eq!(held.len(), HELD_PER_STRATUM);
                    assert_eq!(ext.len(), EXT_PER_STRATUM);
                    let all: BTreeSet<_> = states.iter().copied().collect();
                    assert_eq!(all.len(), PER_STRATUM);
                }
            }
        }
    }

    #[test]
    fn replaying_a_development_key_into_heldout_is_rejected() {
        let mut rows = independent_schedule(V2PublicFamily::PublicFlowV2).unwrap();
        let dev = rows
            .iter()
            .find(|row| row.partition == AuditPartition::Development)
            .unwrap()
            .clone();
        let held_index = rows
            .iter()
            .position(|row| row.partition == AuditPartition::HeldOut)
            .unwrap();
        rows[held_index].pre = dev.pre;
        rows[held_index].action = dev.action;
        rows[held_index].post = dev.post;
        assert_eq!(
            validate_sufficiency_and_novelty(&rows),
            Err(AuditError::CrossPartitionInputOverlap)
        );
    }

    #[test]
    fn ambiguous_public_dynamics_are_rejected() {
        let mut rows = independent_schedule(V2PublicFamily::PublicRelayV2).unwrap();
        let source = rows[0].clone();
        let index = 1;
        rows[index].family = source.family;
        rows[index].pre = source.pre;
        rows[index].action = source.action;
        rows[index].post = AuditState([
            source.post.0[0].saturating_add(7),
            source.post.0[1],
            source.post.0[2],
            source.post.0[3],
        ]);
        assert_eq!(
            validate_sufficiency_and_novelty(&rows),
            Err(AuditError::PublicSufficiency)
        );
    }

    #[test]
    fn actual_post_state_outside_public_schema_is_rejected() {
        let mut rows = independent_schedule(V2PublicFamily::PublicFlowV2).unwrap();
        rows[0].post.0[0] = i32::from(V2_COUNT_CARDINALITY);
        assert_eq!(
            validate_actual_public_domain(&rows),
            Err(AuditError::PublicSchema)
        );
    }

    #[test]
    fn obvious_partition_coding_is_rejected_by_histogram_gate() {
        let mut rows = independent_schedule(V2PublicFamily::PublicFlowV2).unwrap();
        for row in rows
            .iter_mut()
            .filter(|row| row.partition == AuditPartition::HeldOut)
        {
            row.pre.0[0] = i32::from(PRE_STATE_CARDINALITY - 1);
        }
        assert!(matches!(
            validate_distribution(&rows),
            Err(AuditError::HistogramDistributionDrift)
                | Err(AuditError::SparsePartitionHistogram)
        ));
    }

    #[test]
    fn removing_one_heldout_row_breaks_cardinality() {
        let mut rows = independent_schedule(V2PublicFamily::PublicRelayV2).unwrap();
        let index = rows
            .iter()
            .position(|row| row.partition == AuditPartition::HeldOut)
            .unwrap();
        rows.remove(index);
        assert_eq!(validate_structure(&rows), Err(AuditError::WrongCardinality));
    }
}
