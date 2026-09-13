// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Statistical negative controls for EUREKA-002 V2 construct qualification.
//!
//! These tests use the already-frozen V1 paired analysis engine directly. They
//! demonstrate why V2 construct feasibility cannot be reduced to a point-gap
//! calculation: a finite experiment can clear the practical point margin while
//! its paired confidence interval still fails to support a positive effect.

use super::analysis_plan::{CampaignRowDisposition, EUREKA_002_ANALYSIS_PLAN_V1};
use super::constitution::ScientificDisposition;
use super::cross_family_analysis::{
    AnalysisMetricOutcome, EvidenceFamilyId, PairedAnalysisRow, RawConsequenceCounts,
    analyze_cross_family_v1,
};
use super::hidden_world::CorpusPartition;

fn changed_counts(correct: bool) -> RawConsequenceCounts {
    RawConsequenceCounts {
        field_count: 4,
        actual_changed: 1,
        true_positive_changes: u16::from(correct),
        false_positive_changes: 0,
        missed_changes: u16::from(!correct),
        correct_changed_values: u16::from(correct),
        correct_unchanged_values: 3,
    }
}

fn no_change_counts() -> RawConsequenceCounts {
    RawConsequenceCounts {
        field_count: 4,
        actual_changed: 0,
        true_positive_changes: 0,
        false_positive_changes: 0,
        missed_changes: 0,
        correct_changed_values: 0,
        correct_unchanged_values: 4,
    }
}

/// 64 rows: first 32 change-bearing, final 32 true-no-change.
///
/// On change-bearing rows candidate is right on 17, comparator on a strict
/// subset of 15. Therefore candidate is never worse row-by-row, but the entire
/// advantage is carried by only two discordant rows.
fn sparse_advantage_family(
    family: EvidenceFamilyId,
    row_prefix: u64,
) -> Vec<PairedAnalysisRow> {
    (0..64_u64)
        .map(|index| {
            let (candidate, comparator) = if index < 32 {
                (
                    AnalysisMetricOutcome::Scored(changed_counts(index < 17)),
                    AnalysisMetricOutcome::Scored(changed_counts(index < 15)),
                )
            } else {
                (
                    AnalysisMetricOutcome::Scored(no_change_counts()),
                    AnalysisMetricOutcome::Scored(no_change_counts()),
                )
            };
            PairedAnalysisRow {
                row_identity: row_prefix + index,
                family,
                partition: CorpusPartition::HeldOutEvaluation,
                seed_identity: index,
                disposition: CampaignRowDisposition::ValidScored,
                candidate,
                comparator,
            }
        })
        .collect()
}

#[test]
fn practical_point_margin_can_still_fail_frozen_bootstrap_support() {
    let mut rows = sparse_advantage_family(EvidenceFamilyId::ResourceFlowV1, 100_000);
    rows.extend(sparse_advantage_family(
        EvidenceFamilyId::RelayTriadV1,
        200_000,
    ));

    let result = analyze_cross_family_v1(&rows).expect("well-formed paired mutant evidence");
    let f1 = result
        .overall_changed_field_f1
        .expect("change-bearing rows define F1");
    let value = result
        .overall_changed_value_accuracy
        .expect("change-bearing rows define value accuracy");

    assert!(
        f1.point_delta_bps >= EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps,
        "mutant must clear the preregistered F1 point margin"
    );
    assert!(
        value.point_delta_bps
            >= EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps,
        "mutant must clear the preregistered changed-value point margin"
    );

    // Only two of 64 rows carry the paired advantage. A substantial fraction
    // of bootstrap replicates omit both discordant rows, pinning the lower
    // percentile at zero despite a >5pp aggregate point estimate.
    assert!(f1.ci_lower_bps <= 0);
    assert!(value.ci_lower_bps <= 0);
    assert_eq!(result.base_disposition, ScientificDisposition::Inconclusive);
    assert_eq!(result.final_disposition, ScientificDisposition::Inconclusive);
}

#[test]
fn construct_headroom_is_stricter_than_scientific_promotion_margin() {
    let scientific_f1 = i32::from(EUREKA_002_ANALYSIS_PLAN_V1.changed_field_f1_margin_bps);
    let scientific_value =
        i32::from(EUREKA_002_ANALYSIS_PLAN_V1.changed_value_accuracy_margin_bps);
    const CONSTRUCT_SAFETY_HEADROOM_BPS: i32 = 500;

    assert_eq!(scientific_f1, 500);
    assert_eq!(scientific_value, 500);
    assert_eq!(scientific_f1 + CONSTRUCT_SAFETY_HEADROOM_BPS, 1_000);
    assert_eq!(scientific_value + CONSTRUCT_SAFETY_HEADROOM_BPS, 1_000);

    // A benchmark with only 6.25pp oracle/comparator value headroom could still
    // permit a +5pp scientific point result, but V2A must reject it as an
    // underpowered/fragile instrument because it does not clear the independent
    // +5pp construct safety reserve.
    let fragile_oracle_gap_bps = 625;
    assert!(fragile_oracle_gap_bps >= scientific_value);
    assert!(fragile_oracle_gap_bps < scientific_value + CONSTRUCT_SAFETY_HEADROOM_BPS);
}
