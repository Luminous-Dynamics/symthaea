// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 construct-only prototype.
//!
//! Test-only and target-blind: no production FEP session is created here.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use super::analysis_plan::{CampaignRowDisposition, EUREKA_002_ANALYSIS_PLAN_V1};
use super::baselines::ShortcutBaselineKind;
use super::constitution::ScientificDisposition;
use super::cross_family_analysis::{
    AnalysisMetricOutcome, EvidenceFamilyId, PairedAnalysisRow, RawConsequenceCounts,
    analyze_cross_family_v1,
};
use super::hidden_world::{CorpusPartition, PublicAction, PublicObservation, PublicValue};
use super::promotion::PairedEstimateBps;
use super::v2_evidence_identity::{V2EvidencePartition, canonical_row_identity};
use super::v2_public_schema::{
    V2_ACTION_COUNT, V2_COUNT_CARDINALITY, V2_OBSERVATION_DIM, V2_PUBLIC_MODES_PER_FAMILY,
    V2PublicFamily, V2PublicState, action_from_index, public_schema_commitment,
};

const SCHEDULE_REVISION: &str = "EUREKA.002.V2.CONSTRUCT_SCHEDULE.prototype.v5";
const JOINT_ORDER_REVISION: &str = "EUREKA.002.V2.JOINT_DEVELOPMENT_ORDER.prototype.v2";
/// Headroom between the maximum generated pre-channel value and the maximum
/// canonical public count. This is not a same-field delta bound: Relay may copy
/// a much larger neighboring value into a field.
const MAX_REALIZED_VALUE_HEADROOM: u16 = 2;
const PRE_STATE_CARDINALITY: u16 = V2_COUNT_CARDINALITY - MAX_REALIZED_VALUE_HEADROOM;
const PER_STRATUM: usize = 30;
const DEV_PER_STRATUM: usize = 16;
const CAL_PER_STRATUM: usize = 8;
const HELD_PER_STRATUM: usize = 4;
const EXT_PER_STRATUM: usize = 2;
const STRATA: usize =
    (V2_PUBLIC_MODES_PER_FAMILY as usize) * (V2_ACTION_COUNT as usize);
const ROWS_PER_FAMILY: usize = STRATA * PER_STRATUM;
const CONSTRUCT_HEADROOM_BPS: i32 = 500;
const MAX_PARTITION_MEAN_SPREAD_MILLI: i64 = 8_000;

fn analysis_lane(family: V2PublicFamily) -> EvidenceFamilyId {
    match family {
        V2PublicFamily::PublicFlowV2 => EvidenceFamilyId::ResourceFlowV1,
        V2PublicFamily::PublicRelayV2 => EvidenceFamilyId::RelayTriadV1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Partition {
    Development,
    Calibration,
    HeldOut,
    External,
}

impl Partition {
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
            Self::HeldOut => {
                DEV_PER_STRATUM + CAL_PER_STRATUM
                    ..DEV_PER_STRATUM + CAL_PER_STRATUM + HELD_PER_STRATUM
            }
            Self::External => {
                DEV_PER_STRATUM + CAL_PER_STRATUM + HELD_PER_STRATUM..PER_STRATUM
            }
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

    const fn evidence_partition(self) -> V2EvidencePartition {
        match self {
            Self::Development => V2EvidencePartition::Development,
            Self::Calibration => V2EvidencePartition::Calibration,
            Self::HeldOut => V2EvidencePartition::HeldOut,
            Self::External => V2EvidencePartition::ExternalReplication,
        }
    }
}

/// Construct-local integer vector used for both actual states and unconstrained
/// baseline predictions.
///
/// Actual schedule states are created only through [`State::public`], which
/// validates them against [`V2PublicState`]. Baseline predictions intentionally
/// use [`State::prediction`] so finite out-of-domain guesses remain scoreable as
/// wrong rather than being clipped or rejected by the benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct State([i32; V2_OBSERVATION_DIM]);

impl State {
    fn public(fields: [i32; V2_OBSERVATION_DIM]) -> Result<Self, ConstructError> {
        V2PublicState::new(fields)
            .map(|_| Self(fields))
            .map_err(|_| ConstructError::PublicSchema)
    }

    const fn prediction(fields: [i32; V2_OBSERVATION_DIM]) -> Self {
        Self(fields)
    }

    const fn fields(self) -> [i32; V2_OBSERVATION_DIM] {
        self.0
    }

    fn observation(self) -> PublicObservation {
        PublicObservation {
            step: 0,
            fields: self.0.into_iter().map(PublicValue::Count).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Row {
    family: V2PublicFamily,
    partition: Partition,
    mode: u8,
    action_index: u8,
    action: PublicAction,
    pre: State,
    post: State,
    identity: [u8; 32],
}

impl Row {
    fn changed_fields(&self) -> usize {
        self.pre
            .fields()
            .into_iter()
            .zip(self.post.fields())
            .filter(|(before, after)| before != after)
            .count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstructError {
    PublicSchema,
    StateGenerationBudget,
    Cardinality,
    Imbalance,
    PublicSufficiency,
    CrossPartitionInputOverlap,
    CrossPartitionTransitionOverlap,
    PartitionDistributionDrift,
    MissingChangeRows,
    MissingNoChangeRows,
    NoEligibleComparator,
    ComparatorCoverage,
    F1Headroom,
    ValueHeadroom,
    JointOrder,
    BootstrapAnalysis,
    BootstrapNotSupportCapable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
    total: u32,
    scored: u32,
    change_rows: u32,
    tp: u64,
    fp: u64,
    missed: u64,
    correct_changed: u64,
    actual_changed: u64,
    correct_unchanged: u64,
    actual_unchanged: u64,
}

impl Counts {
    fn empty(total: usize) -> Self {
        Self {
            total: u32::try_from(total).expect("V2 count fits u32"),
            scored: 0,
            change_rows: 0,
            tp: 0,
            fp: 0,
            missed: 0,
            correct_changed: 0,
            actual_changed: 0,
            correct_unchanged: 0,
            actual_unchanged: 0,
        }
    }

    fn coverage_bps(self) -> u16 {
        u16::try_from(ratio_bps(u64::from(self.scored), u64::from(self.total)))
            .expect("coverage basis points fit u16")
    }

    fn f1_fraction(self) -> Option<(u64, u64)> {
        let numerator = 2_u64.saturating_mul(self.tp);
        let denominator = numerator.saturating_add(self.fp).saturating_add(self.missed);
        (denominator > 0).then_some((numerator, denominator))
    }

    fn f1_bps(self) -> Option<i32> {
        self.f1_fraction()
            .map(|(numerator, denominator)| ratio_bps(numerator, denominator))
    }

    fn value_bps(self) -> Option<i32> {
        (self.actual_changed > 0)
            .then(|| ratio_bps(self.correct_changed, self.actual_changed))
    }

    fn unchanged_bps(self) -> Option<i32> {
        (self.actual_unchanged > 0)
            .then(|| ratio_bps(self.correct_unchanged, self.actual_unchanged))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaselineReport {
    kind: ShortcutBaselineKind,
    counts: Counts,
}

impl BaselineReport {
    fn eligible(self) -> bool {
        self.counts.coverage_bps()
            >= EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
            && self.counts.change_rows > 0
            && self.counts.f1_fraction().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FamilyReport {
    family: V2PublicFamily,
    schedule_root: [u8; 32],
    selected: ShortcutBaselineKind,
    calibration: Vec<BaselineReport>,
    heldout: BaselineReport,
    oracle: Counts,
    changed_rows: u32,
    no_change_rows: u32,
    mean_spread_milli: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstructReport {
    families: Vec<FamilyReport>,
    total_rows: u32,
    joint_development_root: [u8; 32],
    oracle_bootstrap_disposition: ScientificDisposition,
    oracle_overall_f1: Option<PairedEstimateBps>,
    oracle_overall_value: Option<PairedEstimateBps>,
}

fn action(index: u8) -> Result<PublicAction, ConstructError> {
    action_from_index(usize::from(index)).map_err(|_| ConstructError::PublicSchema)
}

fn transfer_one(values: &mut [i32; 3], from: usize, to: usize) {
    if values[from] > 0 {
        values[from] -= 1;
        values[to] += 1;
    }
}

fn transition(
    family: V2PublicFamily,
    pre: State,
    action_index: u8,
) -> Result<State, ConstructError> {
    match family {
        V2PublicFamily::PublicFlowV2 => {
            let [x, y, z, mode] = pre.fields();
            let mut values = [x, y, z];
            match action_index {
                0 => {}
                1 => transfer_one(&mut values, 0, 1),
                2 => transfer_one(&mut values, 1, 2),
                3 => transfer_one(&mut values, 2, 0),
                _ => return Err(ConstructError::PublicSchema),
            }
            match mode {
                0 => {}
                1 => values[0] = values[0].saturating_add(1),
                2 => transfer_one(&mut values, 0, 1),
                3 => transfer_one(&mut values, 1, 2),
                _ => return Err(ConstructError::PublicSchema),
            }
            State::public([values[0], values[1], values[2], mode])
        }
        V2PublicFamily::PublicRelayV2 => {
            let [mut x, mut y, mut z, rule] = pre.fields();
            match action_index {
                0 => {}
                1 => x = x.saturating_add(1),
                2 => y = y.saturating_add(1),
                3 => z = z.saturating_add(1),
                _ => return Err(ConstructError::PublicSchema),
            }
            match rule {
                4 => {}
                5 => y = x,
                6 => z = y,
                7 => x = z,
                _ => return Err(ConstructError::PublicSchema),
            }
            State::public([x, y, z, rule])
        }
    }
}

fn schedule(family: V2PublicFamily) -> Result<Vec<Row>, ConstructError> {
    let mut rows = Vec::with_capacity(ROWS_PER_FAMILY);
    for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
        let context = family
            .context(mode)
            .map_err(|_| ConstructError::PublicSchema)?;
        for action_index in 0..V2_ACTION_COUNT {
            let states = stratum_states(family, mode, action_index)?;
            for partition in Partition::ALL {
                for index in partition.range() {
                    let triple = states[index];
                    let pre = State::public([triple[0], triple[1], triple[2], context])?;
                    let post = transition(family, pre, action_index)?;
                    let public_action = action(action_index)?;
                    let identity = row_identity(family, partition, public_action, pre, post)?;
                    rows.push(Row {
                        family,
                        partition,
                        mode,
                        action_index,
                        action: public_action,
                        pre,
                        post,
                        identity,
                    });
                }
            }
        }
    }
    (rows.len() == ROWS_PER_FAMILY)
        .then_some(rows)
        .ok_or(ConstructError::Cardinality)
}

/// Partition is deliberately not an input: one common 30-state stream is
/// created per family × public-context × action, then partitioned by position.
///
/// Generated pre-state counts use `0..PRE_STATE_CARDINALITY`, not the complete
/// public domain. With maximum generated pre-channel value 29, PublicFlowV2 can
/// realize at most `max(pre channels) + 2`, while PublicRelayV2 can realize at
/// most `max(pre channels) + 1`. Therefore every realized count remains inside
/// the canonical `0..=31` range. This is a realized-value bound, not a
/// same-field-delta claim: Relay may copy a much larger neighboring value.
fn stratum_states(
    family: V2PublicFamily,
    mode: u8,
    action_index: u8,
) -> Result<Vec<[i32; 3]>, ConstructError> {
    let mut states = Vec::with_capacity(PER_STRATUM);
    let mut seen = BTreeSet::new();
    for ordinal in 0..4096_u32 {
        let digest = schedule_digest(b"public-state", family, mode, action_index, ordinal);
        let bytes = digest.as_bytes();
        let state = [
            i32::from(u16::from_le_bytes([bytes[0], bytes[1]]) % PRE_STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[2], bytes[3]]) % PRE_STATE_CARDINALITY),
            i32::from(u16::from_le_bytes([bytes[4], bytes[5]]) % PRE_STATE_CARDINALITY),
        ];
        if seen.insert(state) {
            states.push(state);
            if states.len() == PER_STRATUM {
                return Ok(states);
            }
        }
    }
    Err(ConstructError::StateGenerationBudget)
}

fn schedule_digest(
    purpose: &[u8],
    family: V2PublicFamily,
    mode: u8,
    action_index: u8,
    ordinal: u32,
) -> blake3::Hash {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, SCHEDULE_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    encode_bytes(&mut bytes, purpose);
    bytes.push(family.tag());
    bytes.push(mode);
    bytes.push(action_index);
    bytes.extend_from_slice(&ordinal.to_le_bytes());
    blake3::hash(&bytes)
}

fn row_identity(
    family: V2PublicFamily,
    partition: Partition,
    action: PublicAction,
    pre: State,
    post: State,
) -> Result<[u8; 32], ConstructError> {
    let pre = V2PublicState::new(pre.fields()).map_err(|_| ConstructError::PublicSchema)?;
    let post = V2PublicState::new(post.fields()).map_err(|_| ConstructError::PublicSchema)?;
    canonical_row_identity(family, partition.evidence_partition(), pre, action, post)
        .map_err(|_| ConstructError::PublicSchema)
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn partition_rows(rows: &[Row], partition: Partition) -> Vec<Row> {
    rows.iter()
        .filter(|row| row.partition == partition)
        .cloned()
        .collect()
}

fn validate_balance(rows: &[Row]) -> Result<(), ConstructError> {
    if rows.len() != ROWS_PER_FAMILY {
        return Err(ConstructError::Cardinality);
    }
    for partition in Partition::ALL {
        for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
            for action_index in 0..V2_ACTION_COUNT {
                let count = rows
                    .iter()
                    .filter(|row| {
                        row.partition == partition
                            && row.mode == mode
                            && row.action_index == action_index
                    })
                    .count();
                if count != partition.expected_per_stratum() {
                    return Err(ConstructError::Imbalance);
                }
            }
        }
    }
    Ok(())
}

fn validate_novelty(rows: &[Row]) -> Result<(), ConstructError> {
    let mut outcomes = BTreeMap::<(V2PublicFamily, State, u8), State>::new();
    let mut key_partitions = BTreeMap::<(V2PublicFamily, State, u8), Partition>::new();
    let mut transition_partitions =
        BTreeMap::<(V2PublicFamily, State, u8, State), Partition>::new();
    for row in rows {
        let key = (row.family, row.pre, row.action_index);
        if let Some(previous) = outcomes.insert(key, row.post) {
            if previous != row.post {
                return Err(ConstructError::PublicSufficiency);
            }
        }
        if let Some(previous) = key_partitions.insert(key, row.partition) {
            if previous != row.partition {
                return Err(ConstructError::CrossPartitionInputOverlap);
            }
        }
        let transition_key = (row.family, row.pre, row.action_index, row.post);
        if let Some(previous) = transition_partitions.insert(transition_key, row.partition) {
            if previous != row.partition {
                return Err(ConstructError::CrossPartitionTransitionOverlap);
            }
        }
    }
    Ok(())
}

fn max_partition_mean_spread_milli(rows: &[Row]) -> Result<i64, ConstructError> {
    let mut result = 0_i64;
    for field in 0..3_usize {
        let mut means = Vec::new();
        for partition in Partition::ALL {
            let mut sum = 0_i64;
            let mut count = 0_i64;
            for row in rows.iter().filter(|row| row.partition == partition) {
                sum = sum.saturating_add(i64::from(row.pre.fields()[field]));
                count = count.saturating_add(1);
            }
            if count == 0 {
                return Err(ConstructError::Cardinality);
            }
            means.push(sum.saturating_mul(1_000) / count);
        }
        let min = means.iter().min().copied().unwrap_or_default();
        let max = means.iter().max().copied().unwrap_or_default();
        result = result.max(max.saturating_sub(min));
    }
    Ok(result)
}

fn baseline_prediction(
    development: &[Row],
    query: &Row,
    kind: ShortcutBaselineKind,
) -> Option<State> {
    match kind {
        ShortcutBaselineKind::ExactLookup => {
            let mut counts = BTreeMap::<State, u32>::new();
            for row in development.iter().filter(|row| {
                row.family == query.family
                    && row.action_index == query.action_index
                    && row.pre == query.pre
            }) {
                *counts.entry(row.post).or_default() += 1;
            }
            choose_mode_state(&counts)
        }
        ShortcutBaselineKind::NearestTransition => development
            .iter()
            .filter(|row| row.family == query.family && row.action_index == query.action_index)
            .map(|row| (distance(row.pre, query.pre), row.identity, row.post))
            .min_by_key(|(dist, identity, _)| (*dist, *identity))
            .map(|(_, _, post)| post),
        ShortcutBaselineKind::ActionMarginalDelta => {
            let candidates: Vec<_> = development
                .iter()
                .filter(|row| row.family == query.family && row.action_index == query.action_index)
                .collect();
            if candidates.is_empty() {
                return None;
            }
            let mut output = query.pre.fields();
            for (index, slot) in output.iter_mut().enumerate() {
                let mut deltas = BTreeMap::<i32, u32>::new();
                for row in &candidates {
                    let delta = row.post.fields()[index].saturating_sub(row.pre.fields()[index]);
                    *deltas.entry(delta).or_default() += 1;
                }
                *slot = slot.saturating_add(choose_mode_i32(&deltas)?);
            }
            Some(State::prediction(output))
        }
        ShortcutBaselineKind::SimpleMarkov => {
            let mut output = query.pre.fields();
            for (index, slot) in output.iter_mut().enumerate() {
                let mut values = BTreeMap::<i32, u32>::new();
                for row in development.iter().filter(|row| {
                    row.family == query.family
                        && row.action_index == query.action_index
                        && row.pre.fields()[index] == query.pre.fields()[index]
                }) {
                    *values.entry(row.post.fields()[index]).or_default() += 1;
                }
                *slot = choose_mode_i32(&values)?;
            }
            Some(State::prediction(output))
        }
    }
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

fn choose_mode_state(counts: &BTreeMap<State, u32>) -> Option<State> {
    counts
        .iter()
        .max_by(|(left_state, left_count), (right_state, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_state.cmp(left_state))
        })
        .map(|(state, _)| *state)
}

fn distance(left: State, right: State) -> u64 {
    left.fields()
        .into_iter()
        .zip(right.fields())
        .map(|(a, b)| i64::from(a).abs_diff(i64::from(b)))
        .sum()
}

fn score(development: &[Row], evaluation: &[Row], kind: ShortcutBaselineKind) -> Counts {
    let mut counts = Counts::empty(evaluation.len());
    for row in evaluation {
        let Some(predicted) = baseline_prediction(development, row, kind) else {
            continue;
        };
        counts.scored = counts.scored.saturating_add(1);
        accumulate(&mut counts, row.pre, predicted, row.post);
    }
    counts
}

fn oracle_score(rows: &[Row]) -> Counts {
    let mut counts = Counts::empty(rows.len());
    for row in rows {
        counts.scored = counts.scored.saturating_add(1);
        accumulate(&mut counts, row.pre, row.post, row.post);
    }
    counts
}

fn accumulate(counts: &mut Counts, pre: State, predicted: State, actual: State) {
    let mut changed = false;
    for ((before, predicted_value), actual_value) in pre
        .fields()
        .into_iter()
        .zip(predicted.fields())
        .zip(actual.fields())
    {
        let did_change = before != actual_value;
        let predicted_change = before != predicted_value;
        if did_change {
            changed = true;
            counts.actual_changed = counts.actual_changed.saturating_add(1);
        } else {
            counts.actual_unchanged = counts.actual_unchanged.saturating_add(1);
        }
        match (did_change, predicted_change) {
            (true, true) => counts.tp = counts.tp.saturating_add(1),
            (false, true) => counts.fp = counts.fp.saturating_add(1),
            (true, false) => counts.missed = counts.missed.saturating_add(1),
            (false, false) => {}
        }
        if did_change && predicted_value == actual_value {
            counts.correct_changed = counts.correct_changed.saturating_add(1);
        }
        if !did_change && predicted_value == actual_value {
            counts.correct_unchanged = counts.correct_unchanged.saturating_add(1);
        }
    }
    if changed {
        counts.change_rows = counts.change_rows.saturating_add(1);
    }
}

fn raw_counts(pre: State, predicted: State, actual: State) -> RawConsequenceCounts {
    let mut actual_changed = 0_u16;
    let mut tp = 0_u16;
    let mut fp = 0_u16;
    let mut missed = 0_u16;
    let mut correct_changed = 0_u16;
    let mut correct_unchanged = 0_u16;
    for ((before, predicted_value), actual_value) in pre
        .fields()
        .into_iter()
        .zip(predicted.fields())
        .zip(actual.fields())
    {
        let did_change = before != actual_value;
        let predicted_change = before != predicted_value;
        actual_changed += u16::from(did_change);
        match (did_change, predicted_change) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => missed += 1,
            (false, false) => {}
        }
        if did_change && predicted_value == actual_value {
            correct_changed += 1;
        }
        if !did_change && predicted_value == actual_value {
            correct_unchanged += 1;
        }
    }
    RawConsequenceCounts {
        field_count: V2_OBSERVATION_DIM as u16,
        actual_changed,
        true_positive_changes: tp,
        false_positive_changes: fp,
        missed_changes: missed,
        correct_changed_values: correct_changed,
        correct_unchanged_values: correct_unchanged,
    }
}

fn construct_analysis_rows(
    family: V2PublicFamily,
    development: &[Row],
    heldout: &[Row],
    selected: ShortcutBaselineKind,
) -> Vec<PairedAnalysisRow> {
    heldout
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let candidate = AnalysisMetricOutcome::Scored(raw_counts(row.pre, row.post, row.post));
            let comparator = baseline_prediction(development, row, selected)
                .map(|prediction| {
                    AnalysisMetricOutcome::Scored(raw_counts(row.pre, prediction, row.post))
                })
                .unwrap_or(AnalysisMetricOutcome::Abstained);
            let index_u64 = u64::try_from(index).expect("HeldOut index fits u64");
            let family_prefix = u64::from(family.tag()) << 56;
            PairedAnalysisRow {
                row_identity: family_prefix | (index_u64 + 1),
                family: analysis_lane(family),
                partition: CorpusPartition::HeldOutEvaluation,
                seed_identity: index_u64,
                disposition: CampaignRowDisposition::ValidScored,
                candidate,
                comparator,
            }
        })
        .collect()
}

fn comparator_order(left: &BaselineReport, right: &BaselineReport) -> Ordering {
    let (left_num, left_den) = left.counts.f1_fraction().unwrap_or((0, 1));
    let (right_num, right_den) = right.counts.f1_fraction().unwrap_or((0, 1));
    (u128::from(left_num) * u128::from(right_den))
        .cmp(&(u128::from(right_num) * u128::from(left_den)))
        .then_with(|| right.kind.stable_id().cmp(left.kind.stable_id()))
}

fn select(reports: &[BaselineReport]) -> Option<ShortcutBaselineKind> {
    reports
        .iter()
        .copied()
        .filter(|report| report.eligible())
        .max_by(comparator_order)
        .map(|report| report.kind)
}

fn schedule_root(rows: &[Row]) -> [u8; 32] {
    let mut identities: Vec<_> = rows.iter().map(|row| row.identity).collect();
    identities.sort_unstable();
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, b"EUREKA.002.V2.SCHEDULE_ROOT.prototype.v2");
    bytes.extend_from_slice(&public_schema_commitment());
    for identity in identities {
        bytes.extend_from_slice(&identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn joint_development() -> Result<Vec<Row>, ConstructError> {
    let mut flow = partition_rows(
        &schedule(V2PublicFamily::PublicFlowV2)?,
        Partition::Development,
    );
    let mut relay = partition_rows(
        &schedule(V2PublicFamily::PublicRelayV2)?,
        Partition::Development,
    );
    if flow.len() != 256 || relay.len() != 256 {
        return Err(ConstructError::JointOrder);
    }
    flow.sort_by_key(order_key);
    relay.sort_by_key(order_key);
    let mut rows = Vec::with_capacity(512);
    for (flow_row, relay_row) in flow.into_iter().zip(relay) {
        rows.push(flow_row);
        rows.push(relay_row);
    }
    Ok(rows)
}

fn order_key(row: &Row) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, JOINT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&row.identity);
    *blake3::hash(&bytes).as_bytes()
}

fn ordered_root(rows: &[Row]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        b"EUREKA.002.V2.JOINT_DEVELOPMENT_ROOT.prototype.v2",
    );
    bytes.extend_from_slice(&public_schema_commitment());
    for row in rows {
        bytes.extend_from_slice(&row.identity);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn audit() -> Result<ConstructReport, ConstructError> {
    let mut family_reports = Vec::new();
    let mut analysis_rows = Vec::new();
    let mut total_rows = 0_u32;
    for family in V2PublicFamily::ALL {
        let rows = schedule(family)?;
        validate_balance(&rows)?;
        validate_novelty(&rows)?;
        let spread = max_partition_mean_spread_milli(&rows)?;
        if spread > MAX_PARTITION_MEAN_SPREAD_MILLI {
            return Err(ConstructError::PartitionDistributionDrift);
        }
        let changed_rows = u32::try_from(
            rows.iter().filter(|row| row.changed_fields() > 0).count(),
        )
        .expect("V2 count fits u32");
        let row_count = u32::try_from(rows.len()).expect("V2 count fits u32");
        let no_change_rows = row_count.saturating_sub(changed_rows);
        if changed_rows == 0 {
            return Err(ConstructError::MissingChangeRows);
        }
        if no_change_rows == 0 {
            return Err(ConstructError::MissingNoChangeRows);
        }

        let development = partition_rows(&rows, Partition::Development);
        let calibration = partition_rows(&rows, Partition::Calibration);
        let heldout = partition_rows(&rows, Partition::HeldOut);
        let calibration_reports: Vec<_> = ShortcutBaselineKind::ALL
            .into_iter()
            .map(|kind| BaselineReport {
                kind,
                counts: score(&development, &calibration, kind),
            })
            .collect();
        let selected = select(&calibration_reports).ok_or(ConstructError::NoEligibleComparator)?;
        let heldout_report = BaselineReport {
            kind: selected,
            counts: score(&development, &heldout, selected),
        };
        if heldout_report.counts.coverage_bps()
            < EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
        {
            return Err(ConstructError::ComparatorCoverage);
        }
        let oracle = oracle_score(&heldout);
        let f1_gap = oracle
            .f1_bps()
            .zip(heldout_report.counts.f1_bps())
            .map(|(oracle_score, comparator_score)| oracle_score - comparator_score)
            .ok_or(ConstructError::F1Headroom)?;
        let value_gap = oracle
            .value_bps()
            .zip(heldout_report.counts.value_bps())
            .map(|(oracle_score, comparator_score)| oracle_score - comparator_score)
            .ok_or(ConstructError::ValueHeadroom)?;
        let required_f1_gap =
            i32::from(EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps)
                + CONSTRUCT_HEADROOM_BPS;
        let required_value_gap =
            i32::from(EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps)
                + CONSTRUCT_HEADROOM_BPS;
        if f1_gap < required_f1_gap {
            return Err(ConstructError::F1Headroom);
        }
        if value_gap < required_value_gap {
            return Err(ConstructError::ValueHeadroom);
        }

        analysis_rows.extend(construct_analysis_rows(
            family,
            &development,
            &heldout,
            selected,
        ));
        total_rows = total_rows.saturating_add(row_count);
        family_reports.push(FamilyReport {
            family,
            schedule_root: schedule_root(&rows),
            selected,
            calibration: calibration_reports,
            heldout: heldout_report,
            oracle,
            changed_rows,
            no_change_rows,
            mean_spread_milli: spread,
        });
    }

    let bootstrap =
        analyze_cross_family_v1(&analysis_rows).map_err(|_| ConstructError::BootstrapAnalysis)?;
    if bootstrap.final_disposition != ScientificDisposition::Supported {
        return Err(ConstructError::BootstrapNotSupportCapable);
    }
    let joint = joint_development()?;
    Ok(ConstructReport {
        families: family_reports,
        total_rows,
        joint_development_root: ordered_root(&joint),
        oracle_bootstrap_disposition: bootstrap.final_disposition,
        oracle_overall_f1: bootstrap.overall_changed_field_f1,
        oracle_overall_value: bootstrap.overall_changed_value_accuracy,
    })
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
    fn generated_pre_state_margin_keeps_realized_actuals_inside_public_schema() {
        assert_eq!(MAX_REALIZED_VALUE_HEADROOM, 2);
        assert_eq!(PRE_STATE_CARDINALITY, V2_COUNT_CARDINALITY - 2);
        for family in V2PublicFamily::ALL {
            for row in schedule(family).unwrap() {
                assert!(V2PublicState::new(row.pre.fields()).is_ok());
                assert!(V2PublicState::new(row.post.fields()).is_ok());
            }
        }
    }

    #[test]
    fn construct_rows_use_shared_canonical_identity() {
        for family in V2PublicFamily::ALL {
            for row in schedule(family).unwrap().iter().take(64) {
                let pre = V2PublicState::new(row.pre.fields()).unwrap();
                let post = V2PublicState::new(row.post.fields()).unwrap();
                let expected = canonical_row_identity(
                    row.family,
                    row.partition.evidence_partition(),
                    pre,
                    row.action,
                    post,
                )
                .unwrap();
                assert_eq!(row.identity, expected);
            }
        }
    }

    #[test]
    fn exact_partition_counts_and_balance_hold() {
        for family in V2PublicFamily::ALL {
            let rows = schedule(family).unwrap();
            validate_balance(&rows).unwrap();
            for partition in Partition::ALL {
                assert_eq!(
                    rows.iter().filter(|row| row.partition == partition).count(),
                    partition.expected_per_stratum() * STRATA
                );
            }
        }
    }

    #[test]
    fn public_sufficiency_and_semantic_partition_novelty_hold() {
        for family in V2PublicFamily::ALL {
            validate_novelty(&schedule(family).unwrap()).unwrap();
        }
    }

    #[test]
    fn exact_lookup_has_zero_calibration_replay_coverage() {
        for family in V2PublicFamily::ALL {
            let rows = schedule(family).unwrap();
            let development = partition_rows(&rows, Partition::Development);
            let calibration = partition_rows(&rows, Partition::Calibration);
            assert_eq!(
                score(&development, &calibration, ShortcutBaselineKind::ExactLookup)
                    .coverage_bps(),
                0
            );
        }
    }

    #[test]
    fn every_partition_contains_change_and_true_no_change_trials() {
        for family in V2PublicFamily::ALL {
            let rows = schedule(family).unwrap();
            for partition in Partition::ALL {
                let subset = partition_rows(&rows, partition);
                assert!(subset.iter().any(|row| row.changed_fields() == 0));
                assert!(subset.iter().any(|row| row.changed_fields() > 0));
            }
        }
    }

    #[test]
    fn partition_numeric_distributions_are_not_grossly_separated() {
        for family in V2PublicFamily::ALL {
            let spread = max_partition_mean_spread_milli(&schedule(family).unwrap()).unwrap();
            assert!(spread <= MAX_PARTITION_MEAN_SPREAD_MILLI);
        }
    }

    #[test]
    fn joint_development_is_deterministic_and_pair_balanced() {
        let first = joint_development().unwrap();
        let second = joint_development().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 512);
        assert_eq!(ordered_root(&first), ordered_root(&second));
        for pair in first.chunks_exact(2) {
            assert_eq!(pair[0].family, V2PublicFamily::PublicFlowV2);
            assert_eq!(pair[1].family, V2PublicFamily::PublicRelayV2);
            assert_eq!(pair[0].partition, Partition::Development);
            assert_eq!(pair[1].partition, Partition::Development);
        }
    }

    #[test]
    fn construct_capsule_proves_point_and_bootstrap_feasibility() {
        let report = audit().unwrap();
        assert_eq!(report.total_rows, 960);
        assert_eq!(report.families.len(), 2);
        assert_eq!(
            report.oracle_bootstrap_disposition,
            ScientificDisposition::Supported
        );
        let f1 = report.oracle_overall_f1.unwrap();
        let value = report.oracle_overall_value.unwrap();
        assert!(f1.point_delta_bps >= EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps);
        assert!(f1.ci_lower_bps > 0);
        assert!(
            value.point_delta_bps
                >= EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps
        );
        assert!(value.ci_lower_bps > 0);
        for family in report.families {
            assert!(family.changed_rows > 0);
            assert!(family.no_change_rows > 0);
            assert!(family.mean_spread_milli <= MAX_PARTITION_MEAN_SPREAD_MILLI);
            assert!(
                family.heldout.counts.coverage_bps()
                    >= EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
            );
            assert_eq!(family.oracle.f1_bps(), Some(10_000));
            assert_eq!(family.oracle.value_bps(), Some(10_000));
            assert!(
                10_000 - family.heldout.counts.f1_bps().unwrap()
                    >= i32::from(EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps)
                        + CONSTRUCT_HEADROOM_BPS
            );
            assert!(
                10_000 - family.heldout.counts.value_bps().unwrap()
                    >= i32::from(
                        EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps,
                    ) + CONSTRUCT_HEADROOM_BPS
            );
        }
    }

    #[test]
    fn target_visible_shape_consumes_canonical_schema() {
        for family in V2PublicFamily::ALL {
            for row in schedule(family).unwrap().iter().take(32) {
                let observation = row.pre.observation();
                assert_eq!(observation.fields.len(), V2_OBSERVATION_DIM);
                assert!(
                    observation
                        .fields
                        .iter()
                        .all(|value| matches!(value, PublicValue::Count(_)))
                );
                assert_eq!(row.action, action(row.action_index).unwrap());
            }
        }
    }
}
