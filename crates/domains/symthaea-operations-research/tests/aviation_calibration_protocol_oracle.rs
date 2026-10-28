// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Separate integration-level contract checks for calibration split integrity.
//! This file uses public API only and fixes expected outcomes explicitly.

use symthaea_operations_research::aviation_calibration_protocol::{
    audit_no_test_set_tuning, audit_split_integrity, EvaluationPartition as P,
    SplitAuditStatus as S, SplitIntegrityPolicy, SplitRecord, TuningChoice,
};

fn rows() -> Vec<SplitRecord> {
    let split_times = [
        (P::Training, 1_000),
        (P::Calibration, 2_000),
        (P::Validation, 3_000),
        (P::Test, 4_000),
    ];
    let mut out = Vec::new();
    for (partition, epoch) in split_times {
        for i in 0..4 {
            out.push(SplitRecord {
                record_id: format!("{partition:?}-record-{i}"),
                event_cluster_id: format!("{partition:?}-episode-{i}"),
                event_time_epoch_seconds: Some(epoch + i),
                partition,
            });
        }
    }
    out
}

fn policy() -> SplitIntegrityPolicy {
    SplitIntegrityPolicy {
        minimum_clusters_per_partition: 4,
        require_strict_chronological_order: true,
    }
}

#[test]
fn independently_declared_clean_fixture_is_accepted() {
    let report = audit_split_integrity(&rows(), policy());
    assert_eq!(report.status, S::Accept);
    assert_eq!(report.partitions.iter().map(|x| x.row_count).sum::<usize>(), 16);
    assert!(report.partitions.iter().all(|x| x.distinct_event_clusters == 4));
}

#[test]
fn changing_cluster_identity_across_test_and_training_is_rejected() {
    let mut data = rows();
    data[12].event_cluster_id = data[0].event_cluster_id.clone();
    let report = audit_split_integrity(&data, policy());
    assert_eq!(report.status, S::Reject);
}

#[test]
fn time_boundary_violation_is_rejected_even_with_disjoint_clusters() {
    let mut data = rows();
    data[8].event_time_epoch_seconds = Some(1_003);
    let report = audit_split_integrity(&data, policy());
    assert_eq!(report.status, S::Reject);
}

#[test]
fn test_partition_used_for_radius_selection_is_rejected() {
    let report = audit_no_test_set_tuning(&[
        TuningChoice {
            artifact_id: "radius".into(),
            selected_using: P::Test,
        },
        TuningChoice {
            artifact_id: "model".into(),
            selected_using: P::Training,
        },
    ]);
    assert_eq!(report.status, S::Reject);
    assert_eq!(report.offending_artifacts, vec!["radius"]);
}

#[test]
fn validation_calibration_choices_are_allowed_but_test_is_not() {
    let report = audit_no_test_set_tuning(&[
        TuningChoice {
            artifact_id: "radius".into(),
            selected_using: P::Calibration,
        },
        TuningChoice {
            artifact_id: "threshold".into(),
            selected_using: P::Validation,
        },
    ]);
    assert_eq!(report.status, S::Accept);
}
