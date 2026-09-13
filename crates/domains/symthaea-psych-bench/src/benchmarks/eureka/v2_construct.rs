// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 construct qualifier.
//!
//! This module no longer owns campaign schedule generation or shortcut-baseline
//! semantics. It qualifies the canonical target-blind corpus materializer and
//! the frozen comparator/selector that downstream campaign code will actually
//! consume. The intentionally independent schedule/dynamics implementation lives
//! in `v2_construct_independent_audit`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use super::analysis_plan::{CampaignRowDisposition, EUREKA_002_ANALYSIS_PLAN_V1};
use super::baselines::ShortcutBaselineKind;
use super::consequence::{
    ConsequenceMetrics, ConsequencePrediction, ConsequenceScore, PredictionOutcome,
    score_consequence,
};
use super::constitution::ScientificDisposition;
use super::cross_family_analysis::{
    AnalysisMetricOutcome, EvidenceFamilyId, PairedAnalysisRow, RawConsequenceCounts,
    analyze_cross_family_v1,
};
use super::hidden_world::{CorpusPartition, PublicObservation, PublicValue};
use super::promotion::PairedEstimateBps;
use super::v2_corpus_schedule::{
    V2_MAX_REALIZED_VALUE_HEADROOM, V2_PRE_STATE_CARDINALITY, V2_ROWS_PER_FAMILY,
    V2_STRATA_PER_FAMILY, V2ScheduleMaterializationError, V2SchedulePartition, V2ScheduledRow,
    canonical_schedule_root, materialize_all_rows, materialize_canonical_corpora,
    materialize_family_rows,
};
use super::v2_frozen_comparator::V2FrozenComparatorSubject;
use super::v2_public_schema::{
    V2_ACTION_COUNT, V2_COUNT_CARDINALITY, V2_OBSERVATION_DIM, V2_PUBLIC_MODES_PER_FAMILY,
    V2PublicFamily, V2PublicState, public_schema_commitment,
};
use super::v2_selection_authorization::{
    V2ComparatorSelectionOutcome, V2SelectionAuthorizationError, execute_calibration_selection,
};

const JOINT_ORDER_REVISION: &str = "EUREKA.002.V2.JOINT_DEVELOPMENT_ORDER.prototype.v2";
const CONSTRUCT_HEADROOM_BPS: i32 = 500;
const MAX_PARTITION_MEAN_SPREAD_MILLI: i64 = 8_000;

fn analysis_lane(family: V2PublicFamily) -> EvidenceFamilyId {
    match family {
        V2PublicFamily::PublicFlowV2 => EvidenceFamilyId::ResourceFlowV1,
        V2PublicFamily::PublicRelayV2 => EvidenceFamilyId::RelayTriadV1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstructError {
    Materialization(V2ScheduleMaterializationError),
    Selection(V2SelectionAuthorizationError),
    Scoring,
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

impl From<V2ScheduleMaterializationError> for ConstructError {
    fn from(value: V2ScheduleMaterializationError) -> Self {
        Self::Materialization(value)
    }
}

impl From<V2SelectionAuthorizationError> for ConstructError {
    fn from(value: V2SelectionAuthorizationError) -> Self {
        Self::Selection(value)
    }
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
            total: u32::try_from(total).expect("V2 construct row count fits u32"),
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

    fn accumulate(&mut self, metrics: ConsequenceMetrics) {
        self.scored = self.scored.saturating_add(1);
        self.change_rows = self
            .change_rows
            .saturating_add(u32::from(metrics.actual_changed > 0));
        self.tp = self
            .tp
            .saturating_add(u64::try_from(metrics.true_positive_changes).expect("count fits u64"));
        self.fp = self
            .fp
            .saturating_add(u64::try_from(metrics.false_positive_changes).expect("count fits u64"));
        self.missed = self
            .missed
            .saturating_add(u64::try_from(metrics.missed_changes).expect("count fits u64"));
        self.correct_changed = self.correct_changed.saturating_add(
            u64::try_from(metrics.correct_changed_values).expect("count fits u64"),
        );
        self.actual_changed = self
            .actual_changed
            .saturating_add(u64::try_from(metrics.actual_changed).expect("count fits u64"));
        self.correct_unchanged = self.correct_unchanged.saturating_add(
            u64::try_from(metrics.correct_unchanged_values).expect("count fits u64"),
        );
        let unchanged = metrics.field_count.saturating_sub(metrics.actual_changed);
        self.actual_unchanged = self
            .actual_unchanged
            .saturating_add(u64::try_from(unchanged).expect("count fits u64"));
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaselineReport {
    kind: ShortcutBaselineKind,
    counts: Counts,
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
    canonical_schedule_root: [u8; 32],
    joint_development_root: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    oracle_bootstrap_disposition: ScientificDisposition,
    oracle_overall_f1: Option<PairedEstimateBps>,
    oracle_overall_value: Option<PairedEstimateBps>,
}

fn observation(state: V2PublicState) -> PublicObservation {
    PublicObservation {
        step: 0,
        fields: state
            .fields()
            .into_iter()
            .map(PublicValue::Count)
            .collect(),
    }
}

fn oracle_prediction(row: V2ScheduledRow) -> ConsequencePrediction {
    ConsequencePrediction {
        action: row.action(),
        outcome: PredictionOutcome::Predicted {
            fields: row
                .post()
                .fields()
                .into_iter()
                .map(PublicValue::Count)
                .collect(),
        },
    }
}

fn score_prediction(
    row: V2ScheduledRow,
    prediction: &ConsequencePrediction,
) -> Result<ConsequenceScore, ConstructError> {
    score_consequence(&observation(row.pre()), prediction, &observation(row.post()))
        .map_err(|_| ConstructError::Scoring)
}

fn score_subject(
    subject: &V2FrozenComparatorSubject,
    rows: &[V2ScheduledRow],
    kind: ShortcutBaselineKind,
) -> Result<Counts, ConstructError> {
    let mut counts = Counts::empty(rows.len());
    for row in rows.iter().copied() {
        let prediction = subject.predict(kind, row.family(), row.pre(), row.action());
        match score_prediction(row, &prediction)? {
            ConsequenceScore::Scored(metrics) => counts.accumulate(metrics),
            ConsequenceScore::AbstainedInsufficientEvidence
            | ConsequenceScore::OutOfQualifiedDomain => {}
        }
    }
    Ok(counts)
}

fn oracle_score(rows: &[V2ScheduledRow]) -> Result<Counts, ConstructError> {
    let mut counts = Counts::empty(rows.len());
    for row in rows.iter().copied() {
        match score_prediction(row, &oracle_prediction(row))? {
            ConsequenceScore::Scored(metrics) => counts.accumulate(metrics),
            ConsequenceScore::AbstainedInsufficientEvidence
            | ConsequenceScore::OutOfQualifiedDomain => return Err(ConstructError::Scoring),
        }
    }
    Ok(counts)
}

fn partition_rows(
    rows: &[V2ScheduledRow],
    partition: V2SchedulePartition,
) -> Vec<V2ScheduledRow> {
    rows.iter()
        .copied()
        .filter(|row| row.partition() == partition)
        .collect()
}

fn changed_fields(row: V2ScheduledRow) -> usize {
    row.pre()
        .fields()
        .into_iter()
        .zip(row.post().fields())
        .filter(|(before, after)| before != after)
        .count()
}

fn validate_balance(rows: &[V2ScheduledRow]) -> Result<(), ConstructError> {
    if rows.len() != V2_ROWS_PER_FAMILY {
        return Err(ConstructError::Cardinality);
    }
    for partition in V2SchedulePartition::ALL {
        for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
            for action_index in 0..V2_ACTION_COUNT {
                let count = rows
                    .iter()
                    .filter(|row| {
                        row.partition() == partition
                            && row.mode() == mode
                            && row.action_index() == action_index
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

fn validate_novelty(rows: &[V2ScheduledRow]) -> Result<(), ConstructError> {
    let mut outcomes = BTreeMap::<([i32; 4], u8), [i32; 4]>::new();
    let mut input_partitions = BTreeMap::<([i32; 4], u8), V2SchedulePartition>::new();
    let mut transition_partitions =
        BTreeMap::<([i32; 4], u8, [i32; 4]), V2SchedulePartition>::new();

    for row in rows.iter().copied() {
        let key = (row.pre().fields(), row.action_index());
        if let Some(previous) = outcomes.insert(key, row.post().fields()) {
            if previous != row.post().fields() {
                return Err(ConstructError::PublicSufficiency);
            }
        }
        if let Some(previous) = input_partitions.insert(key, row.partition()) {
            if previous != row.partition() {
                return Err(ConstructError::CrossPartitionInputOverlap);
            }
        }
        let transition = (row.pre().fields(), row.action_index(), row.post().fields());
        if let Some(previous) = transition_partitions.insert(transition, row.partition()) {
            if previous != row.partition() {
                return Err(ConstructError::CrossPartitionTransitionOverlap);
            }
        }
    }
    Ok(())
}

fn max_partition_mean_spread_milli(rows: &[V2ScheduledRow]) -> Result<i64, ConstructError> {
    let mut result = 0_i64;
    for field in 0..3_usize {
        let mut means = Vec::new();
        for partition in V2SchedulePartition::ALL {
            let mut sum = 0_i64;
            let mut count = 0_i64;
            for row in rows
                .iter()
                .copied()
                .filter(|row| row.partition() == partition)
            {
                sum = sum.saturating_add(i64::from(row.pre().fields()[field]));
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

fn raw_counts_from_metrics(metrics: ConsequenceMetrics) -> RawConsequenceCounts {
    RawConsequenceCounts {
        field_count: u16::try_from(metrics.field_count).expect("V2 field count fits u16"),
        actual_changed: u16::try_from(metrics.actual_changed).expect("V2 field count fits u16"),
        true_positive_changes: u16::try_from(metrics.true_positive_changes)
            .expect("V2 field count fits u16"),
        false_positive_changes: u16::try_from(metrics.false_positive_changes)
            .expect("V2 field count fits u16"),
        missed_changes: u16::try_from(metrics.missed_changes).expect("V2 field count fits u16"),
        correct_changed_values: u16::try_from(metrics.correct_changed_values)
            .expect("V2 field count fits u16"),
        correct_unchanged_values: u16::try_from(metrics.correct_unchanged_values)
            .expect("V2 field count fits u16"),
    }
}

fn analysis_outcome(
    row: V2ScheduledRow,
    prediction: &ConsequencePrediction,
) -> Result<AnalysisMetricOutcome, ConstructError> {
    Ok(match score_prediction(row, prediction)? {
        ConsequenceScore::Scored(metrics) => {
            AnalysisMetricOutcome::Scored(raw_counts_from_metrics(metrics))
        }
        ConsequenceScore::AbstainedInsufficientEvidence
        | ConsequenceScore::OutOfQualifiedDomain => AnalysisMetricOutcome::Abstained,
    })
}

fn construct_analysis_rows(
    family: V2PublicFamily,
    subject: &V2FrozenComparatorSubject,
    heldout: &[V2ScheduledRow],
    selected: ShortcutBaselineKind,
) -> Result<Vec<PairedAnalysisRow>, ConstructError> {
    heldout
        .iter()
        .copied()
        .enumerate()
        .map(|(index, row)| {
            let candidate = analysis_outcome(row, &oracle_prediction(row))?;
            let comparator_prediction =
                subject.predict(selected, row.family(), row.pre(), row.action());
            let comparator = analysis_outcome(row, &comparator_prediction)?;
            let index_u64 = u64::try_from(index).expect("HeldOut index fits u64");
            let family_prefix = u64::from(family.tag()) << 56;
            Ok(PairedAnalysisRow {
                row_identity: family_prefix | (index_u64 + 1),
                family: analysis_lane(family),
                partition: CorpusPartition::HeldOutEvaluation,
                seed_identity: index_u64,
                disposition: CampaignRowDisposition::ValidScored,
                candidate,
                comparator,
            })
        })
        .collect()
}

fn order_key(row: V2ScheduledRow) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, JOINT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&row.row_identity());
    *blake3::hash(&bytes).as_bytes()
}

fn joint_development() -> Result<Vec<V2ScheduledRow>, ConstructError> {
    let mut flow = partition_rows(
        &materialize_family_rows(V2PublicFamily::PublicFlowV2)?,
        V2SchedulePartition::Development,
    );
    let mut relay = partition_rows(
        &materialize_family_rows(V2PublicFamily::PublicRelayV2)?,
        V2SchedulePartition::Development,
    );
    if flow.len() != 256 || relay.len() != 256 {
        return Err(ConstructError::JointOrder);
    }
    flow.sort_by_key(|row| order_key(*row));
    relay.sort_by_key(|row| order_key(*row));
    let mut rows = Vec::with_capacity(512);
    for (flow_row, relay_row) in flow.into_iter().zip(relay) {
        rows.push(flow_row);
        rows.push(relay_row);
    }
    Ok(rows)
}

fn ordered_root(rows: &[V2ScheduledRow]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        b"EUREKA.002.V2.JOINT_DEVELOPMENT_ROOT.prototype.v2",
    );
    bytes.extend_from_slice(&public_schema_commitment());
    for row in rows {
        bytes.extend_from_slice(&row.row_identity());
    }
    *blake3::hash(&bytes).as_bytes()
}

fn audit() -> Result<ConstructReport, ConstructError> {
    let all_rows = materialize_all_rows()?;
    if all_rows.len() != 2 * V2_ROWS_PER_FAMILY {
        return Err(ConstructError::Cardinality);
    }

    let corpora = materialize_canonical_corpora()?;
    let subject = V2FrozenComparatorSubject::freeze(corpora.development());
    let selection = execute_calibration_selection(
        corpora.development(),
        corpora.calibration(),
        &subject,
    )?;
    let selected = match selection {
        V2ComparatorSelectionOutcome::Selected(authorization) => authorization.selected(),
        V2ComparatorSelectionOutcome::Inconclusive(_) => {
            return Err(ConstructError::NoEligibleComparator);
        }
    };

    let mut family_reports = Vec::new();
    let mut analysis_rows = Vec::new();
    let mut total_rows = 0_u32;

    for family in V2PublicFamily::ALL {
        let rows = materialize_family_rows(family)?;
        validate_balance(&rows)?;
        validate_novelty(&rows)?;
        let spread = max_partition_mean_spread_milli(&rows)?;
        if spread > MAX_PARTITION_MEAN_SPREAD_MILLI {
            return Err(ConstructError::PartitionDistributionDrift);
        }

        let changed_rows = u32::try_from(
            rows.iter().copied().filter(|row| changed_fields(*row) > 0).count(),
        )
        .expect("V2 row count fits u32");
        let row_count = u32::try_from(rows.len()).expect("V2 row count fits u32");
        let no_change_rows = row_count.saturating_sub(changed_rows);
        if changed_rows == 0 {
            return Err(ConstructError::MissingChangeRows);
        }
        if no_change_rows == 0 {
            return Err(ConstructError::MissingNoChangeRows);
        }

        let calibration = partition_rows(&rows, V2SchedulePartition::Calibration);
        let heldout = partition_rows(&rows, V2SchedulePartition::HeldOut);
        let calibration_reports = ShortcutBaselineKind::ALL
            .into_iter()
            .map(|kind| {
                Ok(BaselineReport {
                    kind,
                    counts: score_subject(&subject, &calibration, kind)?,
                })
            })
            .collect::<Result<Vec<_>, ConstructError>>()?;
        let heldout_report = BaselineReport {
            kind: selected,
            counts: score_subject(&subject, &heldout, selected)?,
        };
        if heldout_report.counts.coverage_bps()
            < EUREKA_002_ANALYSIS_PLAN_V1.comparator_min_coverage_bps
        {
            return Err(ConstructError::ComparatorCoverage);
        }

        let oracle = oracle_score(&heldout)?;
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
            &subject,
            &heldout,
            selected,
        )?);
        total_rows = total_rows.saturating_add(row_count);
        family_reports.push(FamilyReport {
            family,
            schedule_root: canonical_schedule_root(&rows),
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
        canonical_schedule_root: corpora.schedule_root(),
        joint_development_root: ordered_root(&joint),
        comparator_subject_commitment: subject.commitment(),
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

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_materializer_keeps_realized_actuals_inside_public_schema() {
        assert_eq!(V2_MAX_REALIZED_VALUE_HEADROOM, 2);
        assert_eq!(
            V2_PRE_STATE_CARDINALITY,
            V2_COUNT_CARDINALITY - V2_MAX_REALIZED_VALUE_HEADROOM
        );
        for row in materialize_all_rows().unwrap() {
            assert!(V2PublicState::new(row.pre().fields()).is_ok());
            assert!(V2PublicState::new(row.post().fields()).is_ok());
        }
    }

    #[test]
    fn exact_partition_counts_and_balance_hold_on_materialized_rows() {
        assert_eq!(V2_STRATA_PER_FAMILY, 16);
        for family in V2PublicFamily::ALL {
            let rows = materialize_family_rows(family).unwrap();
            validate_balance(&rows).unwrap();
            for partition in V2SchedulePartition::ALL {
                assert_eq!(
                    rows.iter()
                        .copied()
                        .filter(|row| row.partition() == partition)
                        .count(),
                    partition.expected_per_stratum() * V2_STRATA_PER_FAMILY
                );
            }
        }
    }

    #[test]
    fn construct_rechecks_public_sufficiency_and_partition_novelty() {
        for family in V2PublicFamily::ALL {
            validate_novelty(&materialize_family_rows(family).unwrap()).unwrap();
        }
    }

    #[test]
    fn exact_lookup_has_zero_calibration_replay_coverage_on_canonical_corpus() {
        let corpora = materialize_canonical_corpora().unwrap();
        let subject = V2FrozenComparatorSubject::freeze(corpora.development());
        for family in V2PublicFamily::ALL {
            let rows = materialize_family_rows(family).unwrap();
            let calibration = partition_rows(&rows, V2SchedulePartition::Calibration);
            let counts = score_subject(&subject, &calibration, ShortcutBaselineKind::ExactLookup)
                .unwrap();
            assert_eq!(counts.coverage_bps(), 0);
        }
    }

    #[test]
    fn every_partition_contains_change_and_true_no_change_trials() {
        for family in V2PublicFamily::ALL {
            let rows = materialize_family_rows(family).unwrap();
            for partition in V2SchedulePartition::ALL {
                let subset = partition_rows(&rows, partition);
                assert!(subset.iter().copied().any(|row| changed_fields(row) == 0));
                assert!(subset.iter().copied().any(|row| changed_fields(row) > 0));
            }
        }
    }

    #[test]
    fn partition_numeric_distributions_are_not_grossly_separated() {
        for family in V2PublicFamily::ALL {
            let spread =
                max_partition_mean_spread_milli(&materialize_family_rows(family).unwrap()).unwrap();
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
            assert_eq!(pair[0].family(), V2PublicFamily::PublicFlowV2);
            assert_eq!(pair[1].family(), V2PublicFamily::PublicRelayV2);
            assert_eq!(pair[0].partition(), V2SchedulePartition::Development);
            assert_eq!(pair[1].partition(), V2SchedulePartition::Development);
        }
    }

    #[test]
    fn construct_capsule_qualifies_actual_materializer_and_comparator_subject() {
        let report = audit().unwrap();
        let corpora = materialize_canonical_corpora().unwrap();
        let subject = V2FrozenComparatorSubject::freeze(corpora.development());

        assert_eq!(report.total_rows, 960);
        assert_eq!(report.families.len(), 2);
        assert_eq!(report.canonical_schedule_root, corpora.schedule_root());
        assert_eq!(report.comparator_subject_commitment, subject.commitment());
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

        let selected = report.families[0].selected;
        assert!(report.families.iter().all(|family| family.selected == selected));
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
    fn target_visible_shape_is_exactly_canonical_public_schema() {
        for row in materialize_all_rows().unwrap().into_iter().take(64) {
            let observation = observation(row.pre());
            assert_eq!(observation.fields.len(), V2_OBSERVATION_DIM);
            assert!(
                observation
                    .fields
                    .iter()
                    .all(|value| matches!(value, PublicValue::Count(_)))
            );
            assert_eq!(observation.fields[3], PublicValue::Count(row.pre().context()));
        }
    }
}
