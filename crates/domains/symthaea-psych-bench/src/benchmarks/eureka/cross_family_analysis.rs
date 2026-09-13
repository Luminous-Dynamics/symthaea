// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-family paired analysis for EUREKA-002R.
//!
//! This is a pure interpretation layer. It does not construct worlds, execute a
//! target/comparator, fit a model, or create fresh evidence. Raw integer counts
//! from already-frozen paired rows are authoritative.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2265>

use std::collections::HashSet;

use super::analysis_plan::{CampaignRowDisposition, EUREKA_002_ANALYSIS_PLAN_V1};
use super::constitution::ScientificDisposition;
use super::promotion::{
    PairedEstimateBps, PromotionEvidenceSummary, evaluate_v1,
};

pub(super) const CROSS_FAMILY_ANALYSIS_REVISION: &str =
    "EUREKA.002R.CROSS_FAMILY_ANALYSIS.v1";
/// At least 95% of fixed bootstrap replicates must define a metric interval.
pub(super) const MIN_VALID_BOOTSTRAP_BPS: u16 = 9_500;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum EvidenceFamilyId {
    ResourceFlowV1,
    RelayTriadV1,
}

impl EvidenceFamilyId {
    pub(super) const ALL: [Self; 2] = [Self::ResourceFlowV1, Self::RelayTriadV1];

    pub(super) const fn stable_id(self) -> &'static str {
        match self {
            Self::ResourceFlowV1 => "EUREKA.FAMILY.RESOURCE_FLOW.v1",
            Self::RelayTriadV1 => "EUREKA.FAMILY.RELAY_TRIAD.v1",
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::ResourceFlowV1 => 1,
            Self::RelayTriadV1 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RawConsequenceCounts {
    pub field_count: u16,
    pub actual_changed: u16,
    pub true_positive_changes: u16,
    pub false_positive_changes: u16,
    pub missed_changes: u16,
    pub correct_changed_values: u16,
    pub correct_unchanged_values: u16,
}

impl RawConsequenceCounts {
    fn valid(self) -> bool {
        if self.field_count == 0 || self.actual_changed > self.field_count {
            return false;
        }
        let unchanged = self.field_count - self.actual_changed;
        self.true_positive_changes
            .saturating_add(self.missed_changes)
            == self.actual_changed
            && self.false_positive_changes <= unchanged
            && self.correct_changed_values <= self.true_positive_changes
            && self.correct_unchanged_values <= unchanged
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnalysisMetricOutcome {
    Scored(RawConsequenceCounts),
    Abstained,
    OutOfDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PairedAnalysisRow {
    pub row_identity: u64,
    pub family: EvidenceFamilyId,
    pub seed_identity: u64,
    pub disposition: CampaignRowDisposition,
    pub candidate: AnalysisMetricOutcome,
    pub comparator: AnalysisMetricOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CrossFamilyAnalysisError {
    Empty,
    MissingRequiredFamily,
    DuplicateRowIdentity,
    DuplicateFamilySeed,
    CandidateDispositionMismatch,
    InvalidRawCounts,
    PairedTruthMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct InvalidRowCounts {
    pub leakage: u32,
    pub lineage: u32,
    pub scorer: u32,
    pub target_execution: u32,
    pub infrastructure: u32,
}

impl InvalidRowCounts {
    fn total(self) -> u32 {
        self.leakage
            .saturating_add(self.lineage)
            .saturating_add(self.scorer)
            .saturating_add(self.target_execution)
            .saturating_add(self.infrastructure)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FamilyAnalysisSummary {
    pub family: EvidenceFamilyId,
    pub row_count: u32,
    pub unique_seed_count: u32,
    pub candidate_coverage_bps: u16,
    pub comparator_coverage_bps: u16,
    pub changed_field_f1: Option<PairedEstimateBps>,
    pub changed_value_accuracy: Option<PairedEstimateBps>,
    pub no_change_preservation_delta_bps: Option<i16>,
    pub invalid_rows: InvalidRowCounts,
    pub valid_f1_bootstrap_replicates: u32,
    pub valid_value_bootstrap_replicates: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CrossFamilyAnalysisReceipt {
    pub revision: &'static str,
    pub analysis_plan_digest: u64,
    pub raw_paired_row_root: u64,
    pub family_summaries: Vec<FamilyAnalysisSummary>,
    pub bootstrap_seed: u64,
    pub bootstrap_resamples: u32,
    pub confidence_level_bps: u16,
    pub minimum_valid_bootstrap_bps: u16,
    pub overall_changed_field_f1: Option<PairedEstimateBps>,
    pub overall_changed_value_accuracy: Option<PairedEstimateBps>,
    pub overall_no_change_preservation_delta_bps: Option<i16>,
    pub promotion_summary: PromotionEvidenceSummary,
    pub base_disposition: ScientificDisposition,
    pub final_disposition: ScientificDisposition,
    pub replay_digest: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct MetricAccumulator {
    candidate_tp: u64,
    candidate_fp: u64,
    candidate_missed: u64,
    comparator_tp: u64,
    comparator_fp: u64,
    comparator_missed: u64,
    candidate_correct_changed: u64,
    comparator_correct_changed: u64,
    changed_truth: u64,
    candidate_correct_no_change: u64,
    comparator_correct_no_change: u64,
    no_change_truth: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PointDeltas {
    f1: Option<i16>,
    value: Option<i16>,
    no_change: Option<i16>,
}

#[derive(Debug, Clone, Copy, Default)]
struct BootstrapPoint {
    f1: Option<i16>,
    value: Option<i16>,
}

pub(super) fn analyze_cross_family_v1(
    rows: &[PairedAnalysisRow],
) -> Result<CrossFamilyAnalysisReceipt, CrossFamilyAnalysisError> {
    if rows.is_empty() {
        return Err(CrossFamilyAnalysisError::Empty);
    }
    validate_rows(rows)?;

    let mut family_rows = Vec::new();
    for family in EvidenceFamilyId::ALL {
        let subset: Vec<_> = rows.iter().copied().filter(|row| row.family == family).collect();
        if subset.is_empty() {
            return Err(CrossFamilyAnalysisError::MissingRequiredFamily);
        }
        family_rows.push((family, subset));
    }

    let plan = EUREKA_002_ANALYSIS_PLAN_V1;
    let mut summaries = Vec::with_capacity(2);
    let mut bootstrap_sets = Vec::with_capacity(2);

    for (family, subset) in &family_rows {
        let point = compute_point_deltas(subset.iter().copied());
        let bootstrap = bootstrap_family(*family, subset, plan.bootstrap_resamples, plan.bootstrap_seed);
        let f1_values: Vec<_> = bootstrap.iter().filter_map(|point| point.f1).collect();
        let value_values: Vec<_> = bootstrap.iter().filter_map(|point| point.value).collect();
        let f1 = point.f1.and_then(|point_delta| {
            bootstrap_interval(
                point_delta,
                &f1_values,
                plan.bootstrap_resamples,
                plan.confidence_level_bps,
            )
        });
        let value = point.value.and_then(|point_delta| {
            bootstrap_interval(
                point_delta,
                &value_values,
                plan.bootstrap_resamples,
                plan.confidence_level_bps,
            )
        });
        summaries.push(build_family_summary(*family, subset, f1, value, point.no_change));
        bootstrap_sets.push(bootstrap);
    }

    let overall_f1_point = mean_defined_i16(
        summaries[0].changed_field_f1.map(|e| e.point_delta_bps),
        summaries[1].changed_field_f1.map(|e| e.point_delta_bps),
    );
    let overall_value_point = mean_defined_i16(
        summaries[0].changed_value_accuracy.map(|e| e.point_delta_bps),
        summaries[1].changed_value_accuracy.map(|e| e.point_delta_bps),
    );
    let overall_no_change = mean_defined_i16(
        summaries[0].no_change_preservation_delta_bps,
        summaries[1].no_change_preservation_delta_bps,
    );

    let mut overall_f1_bootstrap = Vec::new();
    let mut overall_value_bootstrap = Vec::new();
    for index in 0..plan.bootstrap_resamples as usize {
        if let (Some(a), Some(b)) = (bootstrap_sets[0][index].f1, bootstrap_sets[1][index].f1) {
            overall_f1_bootstrap.push(mean_i16(a, b));
        }
        if let (Some(a), Some(b)) = (
            bootstrap_sets[0][index].value,
            bootstrap_sets[1][index].value,
        ) {
            overall_value_bootstrap.push(mean_i16(a, b));
        }
    }

    let overall_f1 = overall_f1_point.and_then(|point| {
        bootstrap_interval(
            point,
            &overall_f1_bootstrap,
            plan.bootstrap_resamples,
            plan.confidence_level_bps,
        )
    });
    let overall_value = overall_value_point.and_then(|point| {
        bootstrap_interval(
            point,
            &overall_value_bootstrap,
            plan.bootstrap_resamples,
            plan.confidence_level_bps,
        )
    });

    let invalid = combine_invalid(&summaries);
    let seed_minimum_met = summaries.iter().all(|summary| {
        summary.unique_seed_count >= u32::from(plan.held_out_seeds_per_family_min)
    });
    let promotion_summary = PromotionEvidenceSummary {
        candidate_coverage_bps: mean_u16(
            summaries[0].candidate_coverage_bps,
            summaries[1].candidate_coverage_bps,
        ),
        comparator_coverage_bps: mean_u16(
            summaries[0].comparator_coverage_bps,
            summaries[1].comparator_coverage_bps,
        ),
        changed_field_f1: overall_f1,
        changed_value_accuracy: overall_value,
        no_change_preservation_delta_bps: overall_no_change,
        world_family_count: summaries.len() as u8,
        held_out_seed_minimum_met_for_every_family: seed_minimum_met,
        leakage_rows: invalid.leakage,
        lineage_invalid_rows: invalid.lineage,
        scorer_failure_rows: invalid.scorer,
        target_execution_failure_rows: invalid.target_execution,
        infrastructure_indeterminate_rows: invalid.infrastructure,
    };
    let base_disposition = evaluate_v1(promotion_summary);
    let final_disposition = apply_consistency_gate(base_disposition, &summaries);

    let mut receipt = CrossFamilyAnalysisReceipt {
        revision: CROSS_FAMILY_ANALYSIS_REVISION,
        analysis_plan_digest: plan.replay_digest(),
        raw_paired_row_root: raw_row_root(rows),
        family_summaries: summaries,
        bootstrap_seed: plan.bootstrap_seed,
        bootstrap_resamples: plan.bootstrap_resamples,
        confidence_level_bps: plan.confidence_level_bps,
        minimum_valid_bootstrap_bps: MIN_VALID_BOOTSTRAP_BPS,
        overall_changed_field_f1: overall_f1,
        overall_changed_value_accuracy: overall_value,
        overall_no_change_preservation_delta_bps: overall_no_change,
        promotion_summary,
        base_disposition,
        final_disposition,
        replay_digest: 0,
    };
    receipt.replay_digest = analysis_receipt_digest(&receipt);
    Ok(receipt)
}

fn validate_rows(rows: &[PairedAnalysisRow]) -> Result<(), CrossFamilyAnalysisError> {
    let mut row_ids = HashSet::new();
    let mut family_seed_ids = HashSet::new();
    for row in rows {
        if !row_ids.insert(row.row_identity) {
            return Err(CrossFamilyAnalysisError::DuplicateRowIdentity);
        }
        if !family_seed_ids.insert((row.family, row.seed_identity)) {
            return Err(CrossFamilyAnalysisError::DuplicateFamilySeed);
        }
        if !candidate_matches_disposition(row.disposition, row.candidate) {
            return Err(CrossFamilyAnalysisError::CandidateDispositionMismatch);
        }
        for outcome in [row.candidate, row.comparator] {
            if let AnalysisMetricOutcome::Scored(counts) = outcome {
                if !counts.valid() {
                    return Err(CrossFamilyAnalysisError::InvalidRawCounts);
                }
            }
        }
        if let (
            AnalysisMetricOutcome::Scored(candidate),
            AnalysisMetricOutcome::Scored(comparator),
        ) = (row.candidate, row.comparator)
        {
            if candidate.field_count != comparator.field_count
                || candidate.actual_changed != comparator.actual_changed
            {
                return Err(CrossFamilyAnalysisError::PairedTruthMismatch);
            }
        }
    }
    Ok(())
}

fn candidate_matches_disposition(
    disposition: CampaignRowDisposition,
    candidate: AnalysisMetricOutcome,
) -> bool {
    match disposition {
        CampaignRowDisposition::ValidScored => matches!(candidate, AnalysisMetricOutcome::Scored(_)),
        CampaignRowDisposition::ValidAbstained => candidate == AnalysisMetricOutcome::Abstained,
        CampaignRowDisposition::ValidOutOfDomain => candidate == AnalysisMetricOutcome::OutOfDomain,
        CampaignRowDisposition::InvalidLeakage
        | CampaignRowDisposition::InvalidLineageMismatch
        | CampaignRowDisposition::ScorerFailure
        | CampaignRowDisposition::TargetExecutionFailure
        | CampaignRowDisposition::InfrastructureIndeterminate => true,
    }
}

fn build_family_summary(
    family: EvidenceFamilyId,
    rows: &[PairedAnalysisRow],
    f1: Option<PairedEstimateBps>,
    value: Option<PairedEstimateBps>,
    no_change: Option<i16>,
) -> FamilyAnalysisSummary {
    let total = rows.len() as u64;
    let valid = |row: &&PairedAnalysisRow| is_valid_disposition(row.disposition);
    let candidate_scored = rows
        .iter()
        .filter(valid)
        .filter(|row| matches!(row.candidate, AnalysisMetricOutcome::Scored(_)))
        .count() as u64;
    let comparator_scored = rows
        .iter()
        .filter(valid)
        .filter(|row| matches!(row.comparator, AnalysisMetricOutcome::Scored(_)))
        .count() as u64;
    let invalid_rows = invalid_counts(rows);
    FamilyAnalysisSummary {
        family,
        row_count: rows.len() as u32,
        unique_seed_count: rows.len() as u32,
        candidate_coverage_bps: ratio_u16_bps(candidate_scored, total),
        comparator_coverage_bps: ratio_u16_bps(comparator_scored, total),
        changed_field_f1: f1,
        changed_value_accuracy: value,
        no_change_preservation_delta_bps: no_change,
        invalid_rows,
        valid_f1_bootstrap_replicates: f1
            .map(|_| EUREKA_002_ANALYSIS_PLAN_V1.bootstrap_resamples)
            .unwrap_or(0),
        valid_value_bootstrap_replicates: value
            .map(|_| EUREKA_002_ANALYSIS_PLAN_V1.bootstrap_resamples)
            .unwrap_or(0),
    }
}

fn invalid_counts(rows: &[PairedAnalysisRow]) -> InvalidRowCounts {
    let mut counts = InvalidRowCounts::default();
    for row in rows {
        match row.disposition {
            CampaignRowDisposition::InvalidLeakage => counts.leakage += 1,
            CampaignRowDisposition::InvalidLineageMismatch => counts.lineage += 1,
            CampaignRowDisposition::ScorerFailure => counts.scorer += 1,
            CampaignRowDisposition::TargetExecutionFailure => counts.target_execution += 1,
            CampaignRowDisposition::InfrastructureIndeterminate => counts.infrastructure += 1,
            CampaignRowDisposition::ValidScored
            | CampaignRowDisposition::ValidAbstained
            | CampaignRowDisposition::ValidOutOfDomain => {}
        }
    }
    counts
}

fn combine_invalid(summaries: &[FamilyAnalysisSummary]) -> InvalidRowCounts {
    let mut total = InvalidRowCounts::default();
    for summary in summaries {
        total.leakage = total.leakage.saturating_add(summary.invalid_rows.leakage);
        total.lineage = total.lineage.saturating_add(summary.invalid_rows.lineage);
        total.scorer = total.scorer.saturating_add(summary.invalid_rows.scorer);
        total.target_execution = total
            .target_execution
            .saturating_add(summary.invalid_rows.target_execution);
        total.infrastructure = total
            .infrastructure
            .saturating_add(summary.invalid_rows.infrastructure);
    }
    total
}

fn is_valid_disposition(disposition: CampaignRowDisposition) -> bool {
    matches!(
        disposition,
        CampaignRowDisposition::ValidScored
            | CampaignRowDisposition::ValidAbstained
            | CampaignRowDisposition::ValidOutOfDomain
    )
}

fn compute_point_deltas<'a>(rows: impl Iterator<Item = PairedAnalysisRow> + 'a) -> PointDeltas {
    let mut accumulator = MetricAccumulator::default();
    for row in rows {
        accumulate_row(&mut accumulator, row);
    }
    deltas_from_accumulator(accumulator)
}

fn accumulate_row(acc: &mut MetricAccumulator, row: PairedAnalysisRow) {
    if !is_valid_disposition(row.disposition) {
        return;
    }
    let (
        AnalysisMetricOutcome::Scored(candidate),
        AnalysisMetricOutcome::Scored(comparator),
    ) = (row.candidate, row.comparator)
    else {
        return;
    };
    if candidate.field_count != comparator.field_count
        || candidate.actual_changed != comparator.actual_changed
    {
        return;
    }
    if candidate.actual_changed > 0 {
        acc.candidate_tp += u64::from(candidate.true_positive_changes);
        acc.candidate_fp += u64::from(candidate.false_positive_changes);
        acc.candidate_missed += u64::from(candidate.missed_changes);
        acc.comparator_tp += u64::from(comparator.true_positive_changes);
        acc.comparator_fp += u64::from(comparator.false_positive_changes);
        acc.comparator_missed += u64::from(comparator.missed_changes);
        acc.candidate_correct_changed += u64::from(candidate.correct_changed_values);
        acc.comparator_correct_changed += u64::from(comparator.correct_changed_values);
        acc.changed_truth += u64::from(candidate.actual_changed);
    } else {
        acc.candidate_correct_no_change += u64::from(candidate.correct_unchanged_values);
        acc.comparator_correct_no_change += u64::from(comparator.correct_unchanged_values);
        acc.no_change_truth += u64::from(candidate.field_count);
    }
}

fn deltas_from_accumulator(acc: MetricAccumulator) -> PointDeltas {
    let candidate_f1 = f1_bps(acc.candidate_tp, acc.candidate_fp, acc.candidate_missed);
    let comparator_f1 = f1_bps(acc.comparator_tp, acc.comparator_fp, acc.comparator_missed);
    let candidate_value = ratio_i16_bps(acc.candidate_correct_changed, acc.changed_truth);
    let comparator_value = ratio_i16_bps(acc.comparator_correct_changed, acc.changed_truth);
    let candidate_no_change = ratio_i16_bps(acc.candidate_correct_no_change, acc.no_change_truth);
    let comparator_no_change = ratio_i16_bps(acc.comparator_correct_no_change, acc.no_change_truth);
    PointDeltas {
        f1: subtract_defined(candidate_f1, comparator_f1),
        value: subtract_defined(candidate_value, comparator_value),
        no_change: subtract_defined(candidate_no_change, comparator_no_change),
    }
}

fn bootstrap_family(
    family: EvidenceFamilyId,
    rows: &[PairedAnalysisRow],
    resamples: u32,
    seed: u64,
) -> Vec<BootstrapPoint> {
    let mut output = Vec::with_capacity(resamples as usize);
    for replicate in 0..resamples {
        let mut rng = SplitMix64::new(bootstrap_replicate_seed(seed, family, replicate));
        let mut acc = MetricAccumulator::default();
        for _ in 0..rows.len() {
            let index = (rng.next_u64() % rows.len() as u64) as usize;
            accumulate_row(&mut acc, rows[index]);
        }
        let point = deltas_from_accumulator(acc);
        output.push(BootstrapPoint {
            f1: point.f1,
            value: point.value,
        });
    }
    output
}

fn bootstrap_interval(
    point_delta_bps: i16,
    values: &[i16],
    requested_resamples: u32,
    confidence_level_bps: u16,
) -> Option<PairedEstimateBps> {
    let min_valid = ((u64::from(requested_resamples) * u64::from(MIN_VALID_BOOTSTRAP_BPS)
        + 9_999)
        / 10_000) as usize;
    if values.len() < min_valid || values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n_minus_one = sorted.len() - 1;
    let tail_bps = (10_000_u16.saturating_sub(confidence_level_bps)) / 2;
    let lower_index = (usize::from(tail_bps) * n_minus_one) / 10_000;
    let upper_bps = 10_000_usize.saturating_sub(usize::from(tail_bps));
    let upper_index = ((upper_bps * n_minus_one) + 9_999) / 10_000;
    Some(PairedEstimateBps {
        point_delta_bps,
        ci_lower_bps: sorted[lower_index.min(n_minus_one)],
        ci_upper_bps: sorted[upper_index.min(n_minus_one)],
    })
}

fn apply_consistency_gate(
    base: ScientificDisposition,
    families: &[FamilyAnalysisSummary],
) -> ScientificDisposition {
    let plan = EUREKA_002_ANALYSIS_PLAN_V1;
    let floor = -(plan.no_change_preservation_max_regression_bps as i16);
    if families.iter().any(|family| {
        family.candidate_coverage_bps < plan.candidate_min_coverage_bps
            || family.comparator_coverage_bps < plan.comparator_min_coverage_bps
            || family.unique_seed_count < u32::from(plan.held_out_seeds_per_family_min)
            || family.invalid_rows.total() > 0
            || family.changed_field_f1.is_none()
            || family.changed_value_accuracy.is_none()
            || family.no_change_preservation_delta_bps.is_none()
    }) {
        return ScientificDisposition::Inconclusive;
    }

    match base {
        ScientificDisposition::Supported => {
            let consistent = families.iter().all(|family| {
                family.changed_field_f1.unwrap().point_delta_bps > 0
                    && family.changed_value_accuracy.unwrap().point_delta_bps > 0
                    && family.no_change_preservation_delta_bps.unwrap() >= floor
            });
            if consistent { base } else { ScientificDisposition::Mixed }
        }
        ScientificDisposition::Negative => {
            let consistent = families.iter().all(|family| {
                family.changed_field_f1.unwrap().point_delta_bps < 0
                    && family.changed_value_accuracy.unwrap().point_delta_bps < 0
            });
            if consistent { base } else { ScientificDisposition::Mixed }
        }
        ScientificDisposition::Null => {
            let f1_margin = i32::from(plan.changed_field_f1_margin_bps);
            let value_margin = i32::from(plan.changed_value_accuracy_margin_bps);
            let consistent = families.iter().all(|family| {
                i32::from(family.changed_field_f1.unwrap().point_delta_bps).abs() < f1_margin
                    && i32::from(family.changed_value_accuracy.unwrap().point_delta_bps).abs()
                        < value_margin
                    && family.no_change_preservation_delta_bps.unwrap() >= floor
            });
            if consistent { base } else { ScientificDisposition::Mixed }
        }
        ScientificDisposition::Mixed | ScientificDisposition::Inconclusive => base,
    }
}

fn raw_row_root(rows: &[PairedAnalysisRow]) -> u64 {
    let mut ordered = rows.to_vec();
    ordered.sort_by_key(|row| (row.family, row.seed_identity, row.row_identity));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.002r.raw-paired-rows.v1\0");
    bytes.extend_from_slice(&(ordered.len() as u64).to_le_bytes());
    for row in ordered {
        bytes.push(row.family.tag());
        bytes.extend_from_slice(&row.seed_identity.to_le_bytes());
        bytes.extend_from_slice(&row.row_identity.to_le_bytes());
        bytes.push(disposition_tag(row.disposition));
        encode_outcome(&mut bytes, row.candidate);
        encode_outcome(&mut bytes, row.comparator);
    }
    fnv1a64(&bytes)
}

fn analysis_receipt_digest(receipt: &CrossFamilyAnalysisReceipt) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, receipt.revision);
    bytes.extend_from_slice(&receipt.analysis_plan_digest.to_le_bytes());
    bytes.extend_from_slice(&receipt.raw_paired_row_root.to_le_bytes());
    bytes.extend_from_slice(&(receipt.family_summaries.len() as u64).to_le_bytes());
    for family in &receipt.family_summaries {
        encode_family_summary(&mut bytes, family);
    }
    bytes.extend_from_slice(&receipt.bootstrap_seed.to_le_bytes());
    bytes.extend_from_slice(&receipt.bootstrap_resamples.to_le_bytes());
    bytes.extend_from_slice(&receipt.confidence_level_bps.to_le_bytes());
    bytes.extend_from_slice(&receipt.minimum_valid_bootstrap_bps.to_le_bytes());
    encode_estimate(&mut bytes, receipt.overall_changed_field_f1);
    encode_estimate(&mut bytes, receipt.overall_changed_value_accuracy);
    encode_optional_i16(
        &mut bytes,
        receipt.overall_no_change_preservation_delta_bps,
    );
    encode_promotion_summary(&mut bytes, receipt.promotion_summary);
    bytes.push(disposition_result_tag(receipt.base_disposition));
    bytes.push(disposition_result_tag(receipt.final_disposition));
    fnv1a64(&bytes)
}

fn encode_family_summary(bytes: &mut Vec<u8>, family: &FamilyAnalysisSummary) {
    bytes.push(family.family.tag());
    bytes.extend_from_slice(&family.row_count.to_le_bytes());
    bytes.extend_from_slice(&family.unique_seed_count.to_le_bytes());
    bytes.extend_from_slice(&family.candidate_coverage_bps.to_le_bytes());
    bytes.extend_from_slice(&family.comparator_coverage_bps.to_le_bytes());
    encode_estimate(bytes, family.changed_field_f1);
    encode_estimate(bytes, family.changed_value_accuracy);
    encode_optional_i16(bytes, family.no_change_preservation_delta_bps);
    bytes.extend_from_slice(&family.invalid_rows.leakage.to_le_bytes());
    bytes.extend_from_slice(&family.invalid_rows.lineage.to_le_bytes());
    bytes.extend_from_slice(&family.invalid_rows.scorer.to_le_bytes());
    bytes.extend_from_slice(&family.invalid_rows.target_execution.to_le_bytes());
    bytes.extend_from_slice(&family.invalid_rows.infrastructure.to_le_bytes());
    bytes.extend_from_slice(&family.valid_f1_bootstrap_replicates.to_le_bytes());
    bytes.extend_from_slice(&family.valid_value_bootstrap_replicates.to_le_bytes());
}

fn encode_promotion_summary(bytes: &mut Vec<u8>, summary: PromotionEvidenceSummary) {
    bytes.extend_from_slice(&summary.candidate_coverage_bps.to_le_bytes());
    bytes.extend_from_slice(&summary.comparator_coverage_bps.to_le_bytes());
    encode_estimate(bytes, summary.changed_field_f1);
    encode_estimate(bytes, summary.changed_value_accuracy);
    encode_optional_i16(bytes, summary.no_change_preservation_delta_bps);
    bytes.push(summary.world_family_count);
    bytes.push(u8::from(summary.held_out_seed_minimum_met_for_every_family));
    bytes.extend_from_slice(&summary.leakage_rows.to_le_bytes());
    bytes.extend_from_slice(&summary.lineage_invalid_rows.to_le_bytes());
    bytes.extend_from_slice(&summary.scorer_failure_rows.to_le_bytes());
    bytes.extend_from_slice(&summary.target_execution_failure_rows.to_le_bytes());
    bytes.extend_from_slice(&summary.infrastructure_indeterminate_rows.to_le_bytes());
}

fn encode_estimate(bytes: &mut Vec<u8>, estimate: Option<PairedEstimateBps>) {
    match estimate {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.point_delta_bps.to_le_bytes());
            bytes.extend_from_slice(&value.ci_lower_bps.to_le_bytes());
            bytes.extend_from_slice(&value.ci_upper_bps.to_le_bytes());
        }
        None => bytes.push(0),
    }
}

fn encode_optional_i16(bytes: &mut Vec<u8>, value: Option<i16>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        None => bytes.push(0),
    }
}

fn encode_outcome(bytes: &mut Vec<u8>, outcome: AnalysisMetricOutcome) {
    match outcome {
        AnalysisMetricOutcome::Scored(counts) => {
            bytes.push(1);
            bytes.extend_from_slice(&counts.field_count.to_le_bytes());
            bytes.extend_from_slice(&counts.actual_changed.to_le_bytes());
            bytes.extend_from_slice(&counts.true_positive_changes.to_le_bytes());
            bytes.extend_from_slice(&counts.false_positive_changes.to_le_bytes());
            bytes.extend_from_slice(&counts.missed_changes.to_le_bytes());
            bytes.extend_from_slice(&counts.correct_changed_values.to_le_bytes());
            bytes.extend_from_slice(&counts.correct_unchanged_values.to_le_bytes());
        }
        AnalysisMetricOutcome::Abstained => bytes.push(2),
        AnalysisMetricOutcome::OutOfDomain => bytes.push(3),
    }
}

fn disposition_tag(disposition: CampaignRowDisposition) -> u8 {
    match disposition {
        CampaignRowDisposition::ValidScored => 1,
        CampaignRowDisposition::ValidAbstained => 2,
        CampaignRowDisposition::ValidOutOfDomain => 3,
        CampaignRowDisposition::InvalidLeakage => 4,
        CampaignRowDisposition::InvalidLineageMismatch => 5,
        CampaignRowDisposition::ScorerFailure => 6,
        CampaignRowDisposition::TargetExecutionFailure => 7,
        CampaignRowDisposition::InfrastructureIndeterminate => 8,
    }
}

fn disposition_result_tag(disposition: ScientificDisposition) -> u8 {
    match disposition {
        ScientificDisposition::Supported => 1,
        ScientificDisposition::Mixed => 2,
        ScientificDisposition::Null => 3,
        ScientificDisposition::Negative => 4,
        ScientificDisposition::Inconclusive => 5,
    }
}

fn f1_bps(tp: u64, fp: u64, missed: u64) -> Option<i16> {
    let numerator = 2_u64.saturating_mul(tp);
    let denominator = numerator.saturating_add(fp).saturating_add(missed);
    ratio_i16_bps(numerator, denominator)
}

fn ratio_i16_bps(numerator: u64, denominator: u64) -> Option<i16> {
    if denominator == 0 {
        return None;
    }
    let rounded = (u128::from(numerator) * 10_000 + u128::from(denominator) / 2)
        / u128::from(denominator);
    Some(rounded.min(10_000) as i16)
}

fn ratio_u16_bps(numerator: u64, denominator: u64) -> u16 {
    if denominator == 0 {
        return 0;
    }
    (((u128::from(numerator) * 10_000) + u128::from(denominator) / 2)
        / u128::from(denominator))
        .min(10_000) as u16
}

fn subtract_defined(a: Option<i16>, b: Option<i16>) -> Option<i16> {
    Some(a?.saturating_sub(b?))
}

fn mean_defined_i16(a: Option<i16>, b: Option<i16>) -> Option<i16> {
    Some(mean_i16(a?, b?))
}

fn mean_i16(a: i16, b: i16) -> i16 {
    let sum = i32::from(a) + i32::from(b);
    let rounded = if sum >= 0 { (sum + 1) / 2 } else { (sum - 1) / 2 };
    rounded.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn mean_u16(a: u16, b: u16) -> u16 {
    ((u32::from(a) + u32::from(b) + 1) / 2) as u16
}

fn bootstrap_replicate_seed(base: u64, family: EvidenceFamilyId, replicate: u32) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"eureka.002r.bootstrap-replicate.v1\0");
    bytes.extend_from_slice(&base.to_le_bytes());
    bytes.push(family.tag());
    bytes.extend_from_slice(&replicate.to_le_bytes());
    fnv1a64(&bytes)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changed_counts(good: bool) -> RawConsequenceCounts {
        if good {
            RawConsequenceCounts {
                field_count: 3,
                actual_changed: 1,
                true_positive_changes: 1,
                false_positive_changes: 0,
                missed_changes: 0,
                correct_changed_values: 1,
                correct_unchanged_values: 2,
            }
        } else {
            RawConsequenceCounts {
                field_count: 3,
                actual_changed: 1,
                true_positive_changes: 0,
                false_positive_changes: 0,
                missed_changes: 1,
                correct_changed_values: 0,
                correct_unchanged_values: 2,
            }
        }
    }

    fn no_change_counts() -> RawConsequenceCounts {
        RawConsequenceCounts {
            field_count: 3,
            actual_changed: 0,
            true_positive_changes: 0,
            false_positive_changes: 0,
            missed_changes: 0,
            correct_changed_values: 0,
            correct_unchanged_values: 3,
        }
    }

    fn family_rows(
        family: EvidenceFamilyId,
        candidate_good: bool,
        comparator_good: bool,
        row_base: u64,
    ) -> Vec<PairedAnalysisRow> {
        (0..64_u64)
            .map(|index| {
                let (candidate, comparator) = if index % 2 == 0 {
                    (
                        AnalysisMetricOutcome::Scored(changed_counts(candidate_good)),
                        AnalysisMetricOutcome::Scored(changed_counts(comparator_good)),
                    )
                } else {
                    (
                        AnalysisMetricOutcome::Scored(no_change_counts()),
                        AnalysisMetricOutcome::Scored(no_change_counts()),
                    )
                };
                PairedAnalysisRow {
                    row_identity: row_base + index,
                    family,
                    seed_identity: index,
                    disposition: CampaignRowDisposition::ValidScored,
                    candidate,
                    comparator,
                }
            })
            .collect()
    }

    #[test]
    fn two_consistently_positive_families_can_reach_supported() {
        let mut rows = family_rows(EvidenceFamilyId::ResourceFlowV1, true, false, 1_000);
        rows.extend(family_rows(EvidenceFamilyId::RelayTriadV1, true, false, 2_000));
        let result = analyze_cross_family_v1(&rows).unwrap();
        assert_eq!(result.promotion_summary.world_family_count, 2);
        assert!(result.promotion_summary.held_out_seed_minimum_met_for_every_family);
        assert_eq!(result.base_disposition, ScientificDisposition::Supported);
        assert_eq!(result.final_disposition, ScientificDisposition::Supported);
        assert_eq!(result.bootstrap_resamples, 10_000);
    }

    #[test]
    fn positive_negative_cancellation_cannot_masquerade_as_null() {
        let mut rows = family_rows(EvidenceFamilyId::ResourceFlowV1, true, false, 3_000);
        rows.extend(family_rows(EvidenceFamilyId::RelayTriadV1, false, true, 4_000));
        let result = analyze_cross_family_v1(&rows).unwrap();
        assert_eq!(result.base_disposition, ScientificDisposition::Null);
        assert_eq!(result.final_disposition, ScientificDisposition::Mixed);
    }

    #[test]
    fn duplicate_family_seed_fails_closed() {
        let mut rows = family_rows(EvidenceFamilyId::ResourceFlowV1, true, false, 5_000);
        rows.extend(family_rows(EvidenceFamilyId::RelayTriadV1, true, false, 6_000));
        rows[1].seed_identity = rows[0].seed_identity;
        assert_eq!(
            analyze_cross_family_v1(&rows),
            Err(CrossFamilyAnalysisError::DuplicateFamilySeed)
        );
    }

    #[test]
    fn analysis_receipt_is_order_invariant_and_deterministic() {
        let mut rows = family_rows(EvidenceFamilyId::ResourceFlowV1, true, false, 7_000);
        rows.extend(family_rows(EvidenceFamilyId::RelayTriadV1, true, false, 8_000));
        let a = analyze_cross_family_v1(&rows).unwrap();
        rows.reverse();
        let b = analyze_cross_family_v1(&rows).unwrap();
        assert_eq!(a.raw_paired_row_root, b.raw_paired_row_root);
        assert_eq!(a.overall_changed_field_f1, b.overall_changed_field_f1);
        assert_eq!(a.overall_changed_value_accuracy, b.overall_changed_value_accuracy);
        assert_eq!(a.final_disposition, b.final_disposition);
        assert_eq!(a.replay_digest, b.replay_digest);
    }

    #[test]
    fn malformed_raw_counts_are_not_repaired() {
        let mut rows = family_rows(EvidenceFamilyId::ResourceFlowV1, true, false, 9_000);
        rows.extend(family_rows(EvidenceFamilyId::RelayTriadV1, true, false, 10_000));
        rows[0].candidate = AnalysisMetricOutcome::Scored(RawConsequenceCounts {
            field_count: 3,
            actual_changed: 1,
            true_positive_changes: 1,
            false_positive_changes: 0,
            missed_changes: 1,
            correct_changed_values: 1,
            correct_unchanged_values: 2,
        });
        assert_eq!(
            analyze_cross_family_v1(&rows),
            Err(CrossFamilyAnalysisError::InvalidRawCounts)
        );
    }
}
