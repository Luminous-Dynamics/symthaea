// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen promotion semantics for EUREKA-002C.
//!
//! This policy consumes already-computed paired campaign estimates. It does not
//! run a target, select data, or create evidence. Its purpose is to prevent
//! post-result reinterpretation of a valid campaign.

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::constitution::ScientificDisposition;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PairedEstimateBps {
    pub point_delta_bps: i16,
    pub ci_lower_bps: i16,
    pub ci_upper_bps: i16,
}

impl PairedEstimateBps {
    fn valid_interval(self) -> bool {
        self.ci_lower_bps <= self.ci_upper_bps
            && self.ci_lower_bps <= self.point_delta_bps
            && self.point_delta_bps <= self.ci_upper_bps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PromotionEvidenceSummary {
    pub candidate_coverage_bps: u16,
    pub comparator_coverage_bps: u16,
    pub changed_field_f1: Option<PairedEstimateBps>,
    pub changed_value_accuracy: Option<PairedEstimateBps>,
    pub no_change_preservation_delta_bps: Option<i16>,
    pub world_family_count: u8,
    pub held_out_seed_minimum_met_for_every_family: bool,
    pub leakage_rows: u32,
    pub lineage_invalid_rows: u32,
    pub scorer_failure_rows: u32,
    pub target_execution_failure_rows: u32,
    pub infrastructure_indeterminate_rows: u32,
}

/// Resolve only the scientific disposition for the frozen EUREKA-002 claim.
/// Protocol maturity is tracked separately by the EUREKA constitution.
pub(super) fn evaluate_v1(summary: PromotionEvidenceSummary) -> ScientificDisposition {
    let plan = EUREKA_002_ANALYSIS_PLAN_V1;

    if summary.candidate_coverage_bps < plan.candidate_min_coverage_bps
        || summary.comparator_coverage_bps < plan.comparator_min_coverage_bps
        || summary.world_family_count < plan.minimum_world_families
        || !summary.held_out_seed_minimum_met_for_every_family
        || summary.leakage_rows > 0
        || summary.lineage_invalid_rows > 0
        || summary.scorer_failure_rows > 0
        || summary.target_execution_failure_rows > 0
        || summary.infrastructure_indeterminate_rows > 0
    {
        return ScientificDisposition::Inconclusive;
    }

    let Some(f1) = summary.changed_field_f1 else {
        return ScientificDisposition::Inconclusive;
    };
    let Some(value) = summary.changed_value_accuracy else {
        return ScientificDisposition::Inconclusive;
    };
    let Some(no_change_delta) = summary.no_change_preservation_delta_bps else {
        return ScientificDisposition::Inconclusive;
    };
    if !f1.valid_interval() || !value.valid_interval() {
        return ScientificDisposition::Inconclusive;
    }

    let f1_margin = plan.changed_field_f1_margin_bps;
    let value_margin = plan.changed_value_accuracy_margin_bps;
    let no_change_floor = -(plan.no_change_preservation_max_regression_bps as i16);

    // A point estimate that clears a positive practical margin but whose paired
    // confidence interval does not clear zero is explicitly inconclusive.
    if (f1.point_delta_bps >= f1_margin && f1.ci_lower_bps <= 0)
        || (value.point_delta_bps >= value_margin && value.ci_lower_bps <= 0)
    {
        return ScientificDisposition::Inconclusive;
    }

    let f1_support = f1.point_delta_bps >= f1_margin && f1.ci_lower_bps > 0;
    let value_support =
        value.point_delta_bps >= value_margin && value.ci_lower_bps > 0;
    let no_change_noninferior = no_change_delta >= no_change_floor;

    if f1_support && value_support && no_change_noninferior {
        return ScientificDisposition::Supported;
    }

    let f1_negative =
        f1.point_delta_bps <= -f1_margin && f1.ci_upper_bps < 0;
    let value_negative =
        value.point_delta_bps <= -value_margin && value.ci_upper_bps < 0;
    if f1_negative && value_negative {
        return ScientificDisposition::Negative;
    }

    let f1_practically_small = i32::from(f1.point_delta_bps).abs() < i32::from(f1_margin);
    let value_practically_small =
        i32::from(value.point_delta_bps).abs() < i32::from(value_margin);
    if f1_practically_small && value_practically_small && no_change_noninferior {
        return ScientificDisposition::Null;
    }

    ScientificDisposition::Mixed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(point: i16, lower: i16, upper: i16) -> PairedEstimateBps {
        PairedEstimateBps {
            point_delta_bps: point,
            ci_lower_bps: lower,
            ci_upper_bps: upper,
        }
    }

    fn valid_summary() -> PromotionEvidenceSummary {
        PromotionEvidenceSummary {
            candidate_coverage_bps: 9_500,
            comparator_coverage_bps: 9_500,
            changed_field_f1: Some(estimate(700, 200, 1_100)),
            changed_value_accuracy: Some(estimate(650, 100, 1_000)),
            no_change_preservation_delta_bps: Some(-100),
            world_family_count: 2,
            held_out_seed_minimum_met_for_every_family: true,
            leakage_rows: 0,
            lineage_invalid_rows: 0,
            scorer_failure_rows: 0,
            target_execution_failure_rows: 0,
            infrastructure_indeterminate_rows: 0,
        }
    }

    #[test]
    fn all_frozen_positive_gates_resolve_supported() {
        assert_eq!(
            evaluate_v1(valid_summary()),
            ScientificDisposition::Supported
        );
    }

    #[test]
    fn point_margin_without_interval_support_is_inconclusive() {
        let mut summary = valid_summary();
        summary.changed_field_f1 = Some(estimate(700, -50, 1_100));
        assert_eq!(
            evaluate_v1(summary),
            ScientificDisposition::Inconclusive
        );
    }

    #[test]
    fn insufficient_candidate_coverage_is_inconclusive() {
        let mut summary = valid_summary();
        summary.candidate_coverage_bps = 8_999;
        assert_eq!(
            evaluate_v1(summary),
            ScientificDisposition::Inconclusive
        );
    }

    #[test]
    fn any_leakage_or_lineage_invalidity_blocks_support() {
        let mut leakage = valid_summary();
        leakage.leakage_rows = 1;
        assert_eq!(
            evaluate_v1(leakage),
            ScientificDisposition::Inconclusive
        );

        let mut lineage = valid_summary();
        lineage.lineage_invalid_rows = 1;
        assert_eq!(
            evaluate_v1(lineage),
            ScientificDisposition::Inconclusive
        );
    }

    #[test]
    fn practical_null_is_first_class() {
        let mut summary = valid_summary();
        summary.changed_field_f1 = Some(estimate(100, -100, 300));
        summary.changed_value_accuracy = Some(estimate(50, -150, 250));
        assert_eq!(evaluate_v1(summary), ScientificDisposition::Null);
    }

    #[test]
    fn replicated_material_underperformance_is_negative() {
        let mut summary = valid_summary();
        summary.changed_field_f1 = Some(estimate(-700, -1_000, -200));
        summary.changed_value_accuracy = Some(estimate(-600, -900, -100));
        assert_eq!(evaluate_v1(summary), ScientificDisposition::Negative);
    }

    #[test]
    fn conflicting_primary_gates_resolve_mixed() {
        let mut summary = valid_summary();
        summary.changed_value_accuracy = Some(estimate(100, -100, 300));
        assert_eq!(evaluate_v1(summary), ScientificDisposition::Mixed);
    }

    #[test]
    fn no_change_regression_prevents_supported_promotion() {
        let mut summary = valid_summary();
        summary.no_change_preservation_delta_bps = Some(-201);
        assert_eq!(evaluate_v1(summary), ScientificDisposition::Mixed);
    }

    #[test]
    fn malformed_interval_is_inconclusive_not_repaired() {
        let mut summary = valid_summary();
        summary.changed_field_f1 = Some(estimate(700, 800, 900));
        assert_eq!(
            evaluate_v1(summary),
            ScientificDisposition::Inconclusive
        );
    }

    #[test]
    fn infrastructure_indeterminacy_cannot_be_silently_dropped() {
        let mut summary = valid_summary();
        summary.infrastructure_indeterminate_rows = 1;
        assert_eq!(
            evaluate_v1(summary),
            ScientificDisposition::Inconclusive
        );
    }
}
