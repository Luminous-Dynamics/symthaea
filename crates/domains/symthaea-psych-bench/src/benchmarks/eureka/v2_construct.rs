// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 construct-only prototype.
//!
//! This module is intentionally test-only and target-blind. It never creates
//! or queries a production FEP session. It evaluates only benchmark structure,
//! evaluator-public transitions, shortcut baselines, and an evaluator oracle.
//!
//! Prototype-only fields are allowed to be temporarily unused while the V2
//! receipt shape is still being frozen. Remove this allowance when the V2
//! construct capsule becomes an executable evidence surface.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::ShortcutBaselineKind;
use super::hidden_world::{PublicAction, PublicObservation, PublicValue};

const V2_SCHEDULE_REVISION: &str = "EUREKA.002.V2.CONSTRUCT_SCHEDULE.prototype.v2";
const V2_JOINT_ORDER_REVISION: &str = "EUREKA.002.V2.JOINT_DEVELOPMENT_ORDER.prototype.v1";
const STATE_CARDINALITY: u16 = 32;
const MODES_PER_FAMILY: u8 = 4;
const ACTIONS_PER_FAMILY: u8 = 4;
const STRATA_PER_FAMILY: usize = 16;
const STATES_PER_STRATUM: usize = 30;
const DEVELOPMENT_PER_STRATUM: usize = 16;
const CALIBRATION_PER_STRATUM: usize = 8;
const HELDOUT_PER_STRATUM: usize = 4;
const EXTERNAL_PER_STRATUM: usize = 2;
const ROWS_PER_FAMILY: usize = STRATA_PER_FAMILY * STATES_PER_STRATUM;
const CONSTRUCT_SAFETY_HEADROOM_BPS: i32 = 500;
/// Very loose anti-partition-coding sanity bound over the 0..=31 generated
/// state range. Partition is not an input to state generation; this additionally
/// catches accidental future changes that create gross numeric separation.
const MAX_PARTITION_MEAN_SPREAD_MILLI: i64 = 8_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum V2Family {
    ConservedFlow,
    PublicRelay,
}

impl V2Family {
    const ALL: [Self; 2] = [Self::ConservedFlow, Self::PublicRelay];

    const fn tag(self) -> u8 {
        match self {
            Self::ConservedFlow => 1,
            Self::PublicRelay => 2,
        }
    }

    const fn context_value(self, mode: u8) -> i32 {
        match self {
            Self::ConservedFlow => i32::from(mode),
            Self::PublicRelay => 4 + i32::from(mode),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum V2Partition {
    Development,
    Calibration,
    HeldOut,
    ExternalReplication,
}

impl V2Partition {
    const ALL: [Self; 4] = [
        Self::Development,
        Self::Calibration,
        Self::HeldOut,
        Self::ExternalReplication,
    ];

    fn range(self) -> std::ops::Range<usize> {
        match self {
            Self::Development => 0..DEVELOPMENT_PER_STRATUM,
            Self::Calibration => DEVELOPMENT_PER_STRATUM
                ..DEVELOPMENT_PER_STRATUM + CALIBRATION_PER_STRATUM,
            Self::HeldOut => DEVELOPMENT_PER_STRATUM + CALIBRATION_PER_STRATUM
                ..DEVELOPMENT_PER_STRATUM + CALIBRATION_PER_STRATUM + HELDOUT_PER_STRATUM,
            Self::ExternalReplication => DEVELOPMENT_PER_STRATUM
                + CALIBRATION_PER_STRATUM
                + HELDOUT_PER_STRATUM
                ..STATES_PER_STRATUM,
        }
    }

    const fn expected_per_stratum(self) -> usize {
        match self {
            Self::Development => DEVELOPMENT_PER_STRATUM,
            Self::Calibration => CALIBRATION_PER_STRATUM,
            Self::HeldOut => HELDOUT_PER_STRATUM,
            Self::ExternalReplication => EXTERNAL_PER_STRATUM,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::Development => 1,
            Self::Calibration => 2,
            Self::HeldOut => 3,
            Self::ExternalReplication => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CountState([i32; 4]);

impl CountState {
    fn observation(self, step: u32) -> PublicObservation {
        PublicObservation {
            step,
            fields: self.0.into_iter().map(PublicValue::Count).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct V2Row {
    family: V2Family,
    partition: V2Partition,
    mode: u8,
    pre: CountState,
    action_index: u8,
    action: PublicAction,
    post: CountState,
    row_identity: [u8; 32],
}

impl V2Row {
    fn changed_field_count(&self) -> usize {
        self.pre
            .0
            .into_iter()
            .zip(self.post.0)
            .filter(|(before, after)| before != after)
            .count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstructError {
    StateGenerationBudgetExceeded,
    WrongScheduleCardinality,
    StratumImbalance,
    PublicSufficiencyViolation,
    CrossPartitionInputOverlap,
    CrossPartitionTransitionOverlap,
    PartitionDistributionDrift,
    MissingChangeBearingRows,
    MissingTrueNoChangeRows,
    NoEligibleComparator,
    ComparatorHeldOutCoverageTooLow,
    F1HeadroomTooSmall,
    ChangedValueHeadroomTooSmall,
    JointDevelopmentCardinality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AggregateCounts {
    total_rows: u32,
    scored_rows: u32,
    change_bearing_rows: u32,
    true_positive_changes: u64,
    false_positive_changes: u64,
    missed_changes: u64,
    correct_changed_values: u64,
    actual_changed_values: u64,
    correct_unchanged_values: u64,
    actual_unchanged_values: u64,
}

impl AggregateCounts {
    fn coverage_bps(self) -> u16 {
        u16::try_from(ratio_bps(
            u64::from(self.scored_rows),
            u64::from(self.total_rows),
        ))
        .expect("coverage basis points fit u16")
    }

    fn f1_fraction(self) -> Option<(u64, u64)> {
        let numerator = 2_u64.saturating_mul(self.true_positive_changes);
        let denominator = numerator
            .saturating_add(self.false_positive_changes)
            .saturating_add(self.missed_changes);
        (denominator > 0).then_some((numerator, denominator))
    }

    fn changed_field_f1_bps(self) -> Option<i32> {
        self.f1_fraction()
            .map(|(numerator, denominator)| ratio_bps(numerator, denominator))
    }

    fn changed_value_accuracy_bps(self) -> Option<i32> {
        (self.actual_changed_values > 0)
            .then(|| ratio_bps(self.correct_changed_values, self.actual_changed_values))
    }

    fn unchanged_preservation_bps(self) -> Option<i32> {
        (self.actual_unchanged_values > 0).then(|| {
            ratio_bps(
                self.correct_unchanged_values,
                self.actual_unchanged_values,
            )
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaselineReport {
    kind: ShortcutBaselineKind,
    counts: AggregateCounts,
}

impl BaselineReport {
    fn eligible_for_selection(self) -> bool {
        self.counts.coverage_bps()
            >= EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
            && self.counts.change_bearing_rows > 0
            && self.counts.f1_fraction().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FamilyConstructReport {
    family: V2Family,
    schedule_root: [u8; 32],
    selected_comparator: ShortcutBaselineKind,
    calibration: Vec<BaselineReport>,
    heldout_selected: BaselineReport,
    oracle_heldout: AggregateCounts,
    changed_rows: u32,
    no_change_rows: u32,
    cross_partition_input_overlap: u32,
    cross_partition_transition_overlap: u32,
    max_partition_mean_spread_milli: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct V2ConstructReport {
    schedule_revision: &'static str,
    families: Vec<FamilyConstructReport>,
    total_rows: u32,
    joint_development_rows: u32,
    joint_development_root: [u8; 32],
}

fn canonical_action(index: u8) -> PublicAction {
    match index {
        0 => PublicAction::NoOp,
        1 => PublicAction::Pulse { slot: 0 },
        2 => PublicAction::Pulse { slot: 1 },
        3 => PublicAction::Pulse { slot: 2 },
        _ => unreachable!("V2 action index must be in 0..4"),
    }
}

fn transition(family: V2Family, pre: CountState, action_index: u8) -> CountState {
    match family {
        V2Family::ConservedFlow => transition_conserved_flow(pre, action_index),
        V2Family::PublicRelay => transition_public_relay(pre, action_index),
    }
}

fn transfer_one(values: &mut [i32; 3], from: usize, to: usize) {
    if values[from] > 0 {
        values[from] -= 1;
        values[to] += 1;
    }
}

fn transition_conserved_flow(pre: CountState, action_index: u8) -> CountState {
    let [x, y, z, mode] = pre.0;
    let mut values = [x, y, z];
    match action_index {
        0 => {}
        1 => transfer_one(&mut values, 0, 1),
        2 => transfer_one(&mut values, 1, 2),
        3 => transfer_one(&mut values, 2, 0),
        _ => unreachable!(),
    }
    match mode {
        0 => {}
        1 => values[0] = values[0].saturating_add(1),
        2 => transfer_one(&mut values, 0, 1),
        3 => transfer_one(&mut values, 1, 2),
        _ => unreachable!("ConservedFlow public mode must be in 0..4"),
    }
    CountState([values[0], values[1], values[2], mode])
}

fn transition_public_relay(pre: CountState, action_index: u8) -> CountState {
    let [mut x, mut y, mut z, rule] = pre.0;
    match action_index {
        0 => {}
        1 => x = x.saturating_add(1),
        2 => y = y.saturating_add(1),
        3 => z = z.saturating_add(1),
        _ => unreachable!(),
    }
    match rule {
        4 => {}
        5 => y = x,
        6 => z = y,
        7 => x = z,
        _ => unreachable!("PublicRelay public rule must be in 4..8"),
    }
    CountState([x, y, z, rule])
}

fn schedule_for_family(family: V2Family) -> Result<Vec<V2Row>, ConstructError> {
    let mut rows = Vec::with_capacity(ROWS_PER_FAMILY);
    for mode in 0..MODES_PER_FAMILY {
        for action_index in 0..ACTIONS_PER_FAMILY {
            let states = unique_stratum_states(family, mode, action_index)?;
            for partition in V2Partition::ALL {
                for state_index in partition.range() {
                    let triple = states[state_index];
                    let pre = CountState([
                        triple[0],
                        triple[1],
                        triple[2],
                        family.context_value(mode),
                    ]);
                    let post = transition(family, pre, action_index);
                    let action = canonical_action(action_index);
                    rows.push(V2Row {
                        family,
                        partition,
                        mode,
                        pre,
                        action_index,
                        action,
                        post,
                        row_identity: row_identity(
                            family,
                            partition,
                            mode,
                            action_index,
                            pre,
                            post,
                        ),
                    });
                }
            }
        }
    }
    if rows.len() != ROWS_PER_FAMILY {
        return Err(ConstructError::WrongScheduleCardinality);
    }
    Ok(rows)
}

/// Partition is intentionally absent from this generator signature. Partition
/// assignment happens only after one common 30-state stratum stream is fixed.
fn unique_stratum_states(
    family: V2Family,
    mode: u8,
    action_index: u8,
) -> Result<Vec<[i32; 3]>, ConstructError> {
    let mut accepted = Vec::with_capacity(STATES_PER_STRATUM);
    let mut seen = BTreeSet::new();
    for candidate_ordinal in 0..4096_u32 {
        let digest = mixed_digest(
            b"public-state",
            family,
            mode,
            action_index,
            candidate_ordinal,
        );
        let bytes = digest.as_bytes();
        let triple = [
            i32::from(u16::from_le_bytes([bytes[0], bytes[1]]) % STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[2], bytes[3]]) % STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[4], bytes[5]]) % STATE_CARDINALITY),
        ];
        if seen.insert(triple) {
            accepted.push(triple);
            if accepted.len() == STATES_PER_STRATUM {
                return Ok(accepted);
            }
        }
    }
    Err(ConstructError::StateGenerationBudgetExceeded)
}

fn mixed_digest(
    purpose: &[u8],
    family: V2Family,
    mode: u8,
    action_index: u8,
    candidate_ordinal: u32,
) -> blake3::Hash {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_SCHEDULE_REVISION.as_bytes());
    encode_bytes(&mut bytes, purpose);
    bytes.push(family.tag());
    bytes.push(mode);
    bytes.push(action_index);
    bytes.extend_from_slice(&candidate_ordinal.to_le_bytes());
    blake3::hash(&bytes)
}

fn row_identity(
    family: V2Family,
    partition: V2Partition,
    mode: u8,
    action_index: u8,
    pre: CountState,
    post: CountState,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.ROW.prototype.v2");
    bytes.push(family.tag());
    bytes.push(partition.tag());
    bytes.push(mode);
    bytes.push(action_index);
    encode_state(&mut bytes, pre);
    encode_state(&mut bytes, post);
    *blake3::hash(&bytes).as_bytes()
}

fn family_schedule_root(rows: &[V2Row]) -> [u8; 32] {
    let mut identities: Vec<_> = rows.iter().map(|row| row.row_identity).collect();
    identities.sort_unstable();
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.FAMILY_SCHEDULE_ROOT.prototype.v1");
    for identity in identities {
        bytes.extend_from_slice(&identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn joint_development_order() -> Result<Vec<V2Row>, ConstructError> {
    let mut flow: Vec<_> = schedule_for_family(V2Family::ConservedFlow)?
        .into_iter()
        .filter(|row| row.partition == V2Partition::Development)
        .collect();
    let mut relay: Vec<_> = schedule_for_family(V2Family::PublicRelay)?
        .into_iter()
        .filter(|row| row.partition == V2Partition::Development)
        .collect();
    if flow.len() != 256 || relay.len() != 256 {
        return Err(ConstructError::JointDevelopmentCardinality);
    }
    flow.sort_by_key(development_order_key);
    relay.sort_by_key(development_order_key);
    let mut ordered = Vec::with_capacity(512);
    for (flow_row, relay_row) in flow.into_iter().zip(relay) {
        ordered.push(flow_row);
        ordered.push(relay_row);
    }
    Ok(ordered)
}

fn development_order_key(row: &V2Row) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_JOINT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&row.row_identity);
    *blake3::hash(&bytes).as_bytes()
}

fn joint_development_root(rows: &[V2Row]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.JOINT_DEVELOPMENT_ROOT.prototype.v1");
    for row in rows {
        bytes.extend_from_slice(&row.row_identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn encode_state(bytes: &mut Vec<u8>, state: CountState) {
    for value in state.0 {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn audit_v2_construct() -> Result<V2ConstructReport, ConstructError> {
    let mut family_reports = Vec::new();
    let mut total_rows = 0_u32;

    for family in V2Family::ALL {
        let rows = schedule_for_family(family)?;
        validate_cardinality_and_balance(&rows)?;
        let (input_overlap, transition_overlap) = validate_sufficiency_and_novelty(&rows)?;
        let spread = max_partition_mean_spread_milli(&rows)?;
        if spread > MAX_PARTITION_MEAN_SPREAD_MILLI {
            return Err(ConstructError::PartitionDistributionDrift);
        }

        let changed_rows = u32::try_from(
            rows.iter()
                .filter(|row| row.changed_field_count() > 0)
                .count(),
        )
        .expect("V2 schedule count fits u32");
        let row_count = u32::try_from(rows.len()).expect("V2 schedule count fits u32");
        let no_change_rows = row_count.saturating_sub(changed_rows);
        if changed_rows == 0 {
            return Err(ConstructError::MissingChangeBearingRows);
        }
        if no_change_rows == 0 {
            return Err(ConstructError::MissingTrueNoChangeRows);
        }

        let development = partition_rows(&rows, V2Partition::Development);
        let calibration = partition_rows(&rows, V2Partition::Calibration);
        let heldout = partition_rows(&rows, V2Partition::HeldOut);

        let mut calibration_reports = ShortcutBaselineKind::ALL
            .into_iter()
            .map(|kind| BaselineReport {
                kind,
                counts: score_baseline(&development, &calibration, kind),
            })
            .collect::<Vec<_>>();
        calibration_reports.sort_by(|left, right| left.kind.stable_id().cmp(right.kind.stable_id()));
        let selected = select_comparator(&calibration_reports)
            .ok_or(ConstructError::NoEligibleComparator)?;
        let heldout_selected = BaselineReport {
            kind: selected,
            counts: score_baseline(&development, &heldout, selected),
        };
        if heldout_selected.counts.coverage_bps()
            < EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
        {
            return Err(ConstructError::ComparatorHeldOutCoverageTooLow);
        }

        let oracle_heldout = score_oracle(&heldout);
        let comparator_f1 = heldout_selected
            .counts
            .changed_field_f1_bps()
            .ok_or(ConstructError::F1HeadroomTooSmall)?;
        let oracle_f1 = oracle_heldout
            .changed_field_f1_bps()
            .ok_or(ConstructError::F1HeadroomTooSmall)?;
        let required_f1_gap = i32::from(
            EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps,
        ) + CONSTRUCT_SAFETY_HEADROOM_BPS;
        if oracle_f1 - comparator_f1 < required_f1_gap {
            return Err(ConstructError::F1HeadroomTooSmall);
        }

        let comparator_value = heldout_selected
            .counts
            .changed_value_accuracy_bps()
            .ok_or(ConstructError::ChangedValueHeadroomTooSmall)?;
        let oracle_value = oracle_heldout
            .changed_value_accuracy_bps()
            .ok_or(ConstructError::ChangedValueHeadroomTooSmall)?;
        let required_value_gap = i32::from(
            EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps,
        ) + CONSTRUCT_SAFETY_HEADROOM_BPS;
        if oracle_value - comparator_value < required_value_gap {
            return Err(ConstructError::ChangedValueHeadroomTooSmall);
        }

        total_rows = total_rows.saturating_add(row_count);
        family_reports.push(FamilyConstructReport {
            family,
            schedule_root: family_schedule_root(&rows),
            selected_comparator: selected,
            calibration: calibration_reports,
            heldout_selected,
            oracle_heldout,
            changed_rows,
            no_change_rows,
            cross_partition_input_overlap: input_overlap,
            cross_partition_transition_overlap: transition_overlap,
            max_partition_mean_spread_milli: spread,
        });
    }

    let joint = joint_development_order()?;
    Ok(V2ConstructReport {
        schedule_revision: V2_SCHEDULE_REVISION,
        families: family_reports,
        total_rows,
        joint_development_rows: u32::try_from(joint.len())
            .expect("joint Development count fits u32"),
        joint_development_root: joint_development_root(&joint),
    })
}

fn partition_rows(rows: &[V2Row], partition: V2Partition) -> Vec<V2Row> {
    rows.iter()
        .filter(|row| row.partition == partition)
        .cloned()
        .collect()
}

fn validate_cardinality_and_balance(rows: &[V2Row]) -> Result<(), ConstructError> {
    let Some(family) = rows.first().map(|row| row.family) else {
        return Err(ConstructError::WrongScheduleCardinality);
    };
    if rows.len() != ROWS_PER_FAMILY || rows.iter().any(|row| row.family != family) {
        return Err(ConstructError::WrongScheduleCardinality);
    }
    for partition in V2Partition::ALL {
        for mode in 0..MODES_PER_FAMILY {
            for action in 0..ACTIONS_PER_FAMILY {
                let count = rows
                    .iter()
                    .filter(|row| {
                        row.partition == partition
                            && row.mode == mode
                            && row.action_index == action
                    })
                    .count();
                if count != partition.expected_per_stratum() {
                    return Err(ConstructError::StratumImbalance);
                }
            }
        }
    }
    Ok(())
}

fn validate_sufficiency_and_novelty(rows: &[V2Row]) -> Result<(u32, u32), ConstructError> {
    let mut outcome_by_key = BTreeMap::<(V2Family, CountState, u8), CountState>::new();
    let mut partition_by_key = BTreeMap::<(V2Family, CountState, u8), V2Partition>::new();
    let mut partition_by_transition =
        BTreeMap::<(V2Family, CountState, u8, CountState), V2Partition>::new();
    let mut input_overlap = 0_u32;
    let mut transition_overlap = 0_u32;

    for row in rows {
        let key = (row.family, row.pre, row.action_index);
        if let Some(existing) = outcome_by_key.insert(key, row.post) {
            if existing != row.post {
                return Err(ConstructError::PublicSufficiencyViolation);
            }
        }
        if let Some(existing_partition) = partition_by_key.insert(key, row.partition) {
            if existing_partition != row.partition {
                input_overlap = input_overlap.saturating_add(1);
            }
        }
        let transition_key = (row.family, row.pre, row.action_index, row.post);
        if let Some(existing_partition) =
            partition_by_transition.insert(transition_key, row.partition)
        {
            if existing_partition != row.partition {
                transition_overlap = transition_overlap.saturating_add(1);
            }
        }
    }

    if input_overlap > 0 {
        return Err(ConstructError::CrossPartitionInputOverlap);
    }
    if transition_overlap > 0 {
        return Err(ConstructError::CrossPartitionTransitionOverlap);
    }
    Ok((input_overlap, transition_overlap))
}

fn max_partition_mean_spread_milli(rows: &[V2Row]) -> Result<i64, ConstructError> {
    let mut max_spread = 0_i64;
    for field in 0..3_usize {
        let mut means = Vec::new();
        for partition in V2Partition::ALL {
            let mut sum = 0_i64;
            let mut count = 0_i64;
            for row in rows.iter().filter(|row| row.partition == partition) {
                sum = sum.saturating_add(i64::from(row.pre.0[field]));
                count = count.saturating_add(1);
            }
            if count == 0 {
                return Err(ConstructError::WrongScheduleCardinality);
            }
            means.push(sum.saturating_mul(1_000) / count);
        }
        let min = means.iter().min().copied().unwrap_or_default();
        let max = means.iter().max().copied().unwrap_or_default();
        max_spread = max_spread.max(max.saturating_sub(min));
    }
    Ok(max_spread)
}

fn select_comparator(reports: &[BaselineReport]) -> Option<ShortcutBaselineKind> {
    reports
        .iter()
        .copied()
        .filter(|report| report.eligible_for_selection())
        .max_by(comparator_order)
        .map(|report| report.kind)
}

/// Exact V1-style ordering: compare rational micro-F1 without rounding, then
/// deterministically prefer the lexicographically smaller stable baseline ID.
fn comparator_order(left: &BaselineReport, right: &BaselineReport) -> Ordering {
    let (left_num, left_den) = left.counts.f1_fraction().unwrap_or((0, 1));
    let (right_num, right_den) = right.counts.f1_fraction().unwrap_or((0, 1));
    let f1_order = (u128::from(left_num) * u128::from(right_den))
        .cmp(&(u128::from(right_num) * u128::from(left_den)));
    f1_order.then_with(|| right.kind.stable_id().cmp(left.kind.stable_id()))
}

fn score_baseline(
    development: &[V2Row],
    evaluation: &[V2Row],
    kind: ShortcutBaselineKind,
) -> AggregateCounts {
    let mut counts = empty_counts(evaluation.len());
    for row in evaluation {
        let Some(predicted) = baseline_prediction(development, row, kind) else {
            continue;
        };
        counts.scored_rows = counts.scored_rows.saturating_add(1);
        accumulate_score(&mut counts, row.pre, predicted, row.post);
    }
    counts
}

fn score_oracle(rows: &[V2Row]) -> AggregateCounts {
    let mut oracle = BTreeMap::<(V2Family, CountState, u8), CountState>::new();
    for row in rows {
        oracle.insert((row.family, row.pre, row.action_index), row.post);
    }
    let mut counts = empty_counts(rows.len());
    for row in rows {
        let predicted = *oracle
            .get(&(row.family, row.pre, row.action_index))
            .expect("public-sufficiency audit populated oracle");
        counts.scored_rows = counts.scored_rows.saturating_add(1);
        accumulate_score(&mut counts, row.pre, predicted, row.post);
    }
    counts
}

fn empty_counts(total_rows: usize) -> AggregateCounts {
    AggregateCounts {
        total_rows: u32::try_from(total_rows).expect("V2 row count fits u32"),
        scored_rows: 0,
        change_bearing_rows: 0,
        true_positive_changes: 0,
        false_positive_changes: 0,
        missed_changes: 0,
        correct_changed_values: 0,
        actual_changed_values: 0,
        correct_unchanged_values: 0,
        actual_unchanged_values: 0,
    }
}

fn accumulate_score(
    counts: &mut AggregateCounts,
    pre: CountState,
    predicted: CountState,
    actual: CountState,
) {
    let mut row_changed = false;
    for ((before, predicted_value), actual_value) in
        pre.0.into_iter().zip(predicted.0).zip(actual.0)
    {
        let did_change = before != actual_value;
        let predicted_change = before != predicted_value;
        if did_change {
            row_changed = true;
            counts.actual_changed_values = counts.actual_changed_values.saturating_add(1);
        } else {
            counts.actual_unchanged_values = counts.actual_unchanged_values.saturating_add(1);
        }
        match (did_change, predicted_change) {
            (true, true) => {
                counts.true_positive_changes = counts.true_positive_changes.saturating_add(1)
            }
            (false, true) => {
                counts.false_positive_changes = counts.false_positive_changes.saturating_add(1)
            }
            (true, false) => counts.missed_changes = counts.missed_changes.saturating_add(1),
            (false, false) => {}
        }
        if did_change && predicted_value == actual_value {
            counts.correct_changed_values = counts.correct_changed_values.saturating_add(1);
        }
        if !did_change && predicted_value == actual_value {
            counts.correct_unchanged_values = counts.correct_unchanged_values.saturating_add(1);
        }
    }
    if row_changed {
        counts.change_bearing_rows = counts.change_bearing_rows.saturating_add(1);
    }
}

fn baseline_prediction(
    development: &[V2Row],
    query: &V2Row,
    kind: ShortcutBaselineKind,
) -> Option<CountState> {
    match kind {
        ShortcutBaselineKind::ActionMarginalDelta => action_marginal_delta(development, query),
        ShortcutBaselineKind::ExactLookup => exact_lookup(development, query),
        ShortcutBaselineKind::NearestTransition => nearest_transition(development, query),
        ShortcutBaselineKind::SimpleMarkov => simple_markov(development, query),
    }
}

fn exact_lookup(development: &[V2Row], query: &V2Row) -> Option<CountState> {
    let mut counts = BTreeMap::<CountState, u32>::new();
    for row in development.iter().filter(|row| {
        row.family == query.family && row.action_index == query.action_index && row.pre == query.pre
    }) {
        *counts.entry(row.post).or_default() += 1;
    }
    choose_mode_state(&counts)
}

fn nearest_transition(development: &[V2Row], query: &V2Row) -> Option<CountState> {
    development
        .iter()
        .filter(|row| row.family == query.family && row.action_index == query.action_index)
        .map(|row| (state_distance(row.pre, query.pre), row.row_identity, row.post))
        .min_by_key(|(distance, identity, _)| (*distance, *identity))
        .map(|(_, _, post)| post)
}

fn action_marginal_delta(development: &[V2Row], query: &V2Row) -> Option<CountState> {
    let candidates: Vec<_> = development
        .iter()
        .filter(|row| row.family == query.family && row.action_index == query.action_index)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let mut result = query.pre.0;
    for (index, slot) in result.iter_mut().enumerate() {
        let mut deltas = BTreeMap::<i32, u32>::new();
        for row in &candidates {
            let delta = row.post.0[index].saturating_sub(row.pre.0[index]);
            *deltas.entry(delta).or_default() += 1;
        }
        let delta = choose_mode_i32(&deltas)?;
        *slot = slot.saturating_add(delta);
    }
    Some(CountState(result))
}

fn simple_markov(development: &[V2Row], query: &V2Row) -> Option<CountState> {
    let mut result = query.pre.0;
    for (index, slot) in result.iter_mut().enumerate() {
        let mut outputs = BTreeMap::<i32, u32>::new();
        for row in development.iter().filter(|row| {
            row.family == query.family
                && row.action_index == query.action_index
                && row.pre.0[index] == query.pre.0[index]
        }) {
            *outputs.entry(row.post.0[index]).or_default() += 1;
        }
        *slot = choose_mode_i32(&outputs)?;
    }
    Some(CountState(result))
}

fn choose_mode_i32(counts: &BTreeMap<i32, u32>) -> Option<i32> {
    counts
        .iter()
        .max_by(|(left_value, left_count), (right_value, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_value.cmp(left_value))
        })
        .map(|(value, _)| *value)
}

fn choose_mode_state(counts: &BTreeMap<CountState, u32>) -> Option<CountState> {
    counts
        .iter()
        .max_by(|(left_state, left_count), (right_state, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_state.cmp(left_state))
        })
        .map(|(state, _)| *state)
}

fn state_distance(left: CountState, right: CountState) -> u64 {
    left.0
        .into_iter()
        .zip(right.0)
        .map(|(a, b)| i64::from(a).abs_diff(i64::from(b)))
        .sum()
}

fn ratio_bps(numerator: u64, denominator: u64) -> i32 {
    if denominator == 0 {
        return 0;
    }
    let value = (u128::from(numerator) * 10_000_u128) / u128::from(denominator);
    i32::try_from(value).expect("basis-point ratio fits i32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_has_exact_counts_and_stratified_balance() {
        for family in V2Family::ALL {
            let rows = schedule_for_family(family).unwrap();
            validate_cardinality_and_balance(&rows).unwrap();
            for partition in V2Partition::ALL {
                let count = rows
                    .iter()
                    .filter(|row| row.partition == partition)
                    .count();
                assert_eq!(
                    count,
                    partition.expected_per_stratum() * STRATA_PER_FAMILY
                );
            }
        }
    }

    #[test]
    fn public_input_is_sufficient_and_partitions_are_semantically_disjoint() {
        for family in V2Family::ALL {
            let rows = schedule_for_family(family).unwrap();
            assert_eq!(validate_sufficiency_and_novelty(&rows).unwrap(), (0, 0));
        }
    }

    #[test]
    fn partition_assignment_does_not_create_gross_numeric_range_codes() {
        for family in V2Family::ALL {
            let rows = schedule_for_family(family).unwrap();
            let spread = max_partition_mean_spread_milli(&rows).unwrap();
            assert!(spread <= MAX_PARTITION_MEAN_SPREAD_MILLI);
        }
    }

    #[test]
    fn exact_lookup_cannot_replay_calibration_from_development() {
        for family in V2Family::ALL {
            let rows = schedule_for_family(family).unwrap();
            let development = partition_rows(&rows, V2Partition::Development);
            let calibration = partition_rows(&rows, V2Partition::Calibration);
            let exact = score_baseline(
                &development,
                &calibration,
                ShortcutBaselineKind::ExactLookup,
            );
            assert_eq!(exact.coverage_bps(), 0);
        }
    }

    #[test]
    fn every_partition_has_true_no_change_and_change_bearing_trials() {
        for family in V2Family::ALL {
            let rows = schedule_for_family(family).unwrap();
            for partition in V2Partition::ALL {
                let partition_rows = partition_rows(&rows, partition);
                assert!(partition_rows.iter().any(|row| row.changed_field_count() == 0));
                assert!(partition_rows.iter().any(|row| row.changed_field_count() > 0));
            }
        }
    }

    #[test]
    fn joint_development_order_is_deterministic_and_pair_balanced() {
        let first = joint_development_order().unwrap();
        let second = joint_development_order().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 512);
        assert_eq!(joint_development_root(&first), joint_development_root(&second));
        for pair in first.chunks_exact(2) {
            assert_eq!(pair[0].family, V2Family::ConservedFlow);
            assert_eq!(pair[1].family, V2Family::PublicRelay);
            assert_eq!(pair[0].partition, V2Partition::Development);
            assert_eq!(pair[1].partition, V2Partition::Development);
        }
    }

    #[test]
    fn construct_capsule_proves_comparator_coverage_and_margin_feasibility() {
        let report = audit_v2_construct().unwrap();
        assert_eq!(report.total_rows, 960);
        assert_eq!(report.joint_development_rows, 512);
        assert_eq!(report.families.len(), 2);
        for family in report.families {
            assert_eq!(family.cross_partition_input_overlap, 0);
            assert_eq!(family.cross_partition_transition_overlap, 0);
            assert!(family.changed_rows > 0);
            assert!(family.no_change_rows > 0);
            assert!(family.max_partition_mean_spread_milli <= MAX_PARTITION_MEAN_SPREAD_MILLI);
            assert!(
                family.heldout_selected.counts.coverage_bps()
                    >= EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
            );
            let comparator_f1 = family
                .heldout_selected
                .counts
                .changed_field_f1_bps()
                .unwrap();
            let comparator_value = family
                .heldout_selected
                .counts
                .changed_value_accuracy_bps()
                .unwrap();
            assert_eq!(family.oracle_heldout.changed_field_f1_bps(), Some(10_000));
            assert_eq!(
                family.oracle_heldout.changed_value_accuracy_bps(),
                Some(10_000)
            );
            let required_f1_gap = i32::from(
                EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps,
            ) + CONSTRUCT_SAFETY_HEADROOM_BPS;
            let required_value_gap = i32::from(
                EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps,
            ) + CONSTRUCT_SAFETY_HEADROOM_BPS;
            assert!(10_000 - comparator_f1 >= required_f1_gap);
            assert!(10_000 - comparator_value >= required_value_gap);
        }
    }

    #[test]
    fn target_visible_shape_is_four_counts_and_four_exact_actions() {
        for family in V2Family::ALL {
            let rows = schedule_for_family(family).unwrap();
            for row in rows.iter().take(32) {
                let observation = row.pre.observation(0);
                assert_eq!(observation.fields.len(), 4);
                assert!(
                    observation
                        .fields
                        .iter()
                        .all(|value| matches!(value, PublicValue::Count(_)))
                );
                assert_eq!(row.action, canonical_action(row.action_index));
            }
        }
    }
}
