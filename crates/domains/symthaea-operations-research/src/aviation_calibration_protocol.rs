// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dataset-split integrity gates for aviation robustness calibration.
//!
//! Audits supplied split metadata for duplicate records, event-cluster leakage,
//! undersized partitions, missing event times and optional chronology. It also
//! rejects declared tuning choices that use the final test partition.
//!
//! Passing checks only the metadata presented to this function. It cannot prove
//! that event clusters were correctly defined or that external analysis did not
//! inspect test outcomes.

use std::collections::{BTreeMap, BTreeSet};

/// Partition purpose in a frozen evaluation protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvaluationPartition {
    Training,
    Calibration,
    Validation,
    Test,
}

impl EvaluationPartition {
    fn ordered() -> [Self; 4] {
        [Self::Training, Self::Calibration, Self::Validation, Self::Test]
    }
}

/// One record's split metadata. The event-cluster ID should identify a complete
/// disruption episode or other dependence cluster, not merely an individual row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitRecord {
    pub record_id: String,
    pub event_cluster_id: String,
    pub event_time_epoch_seconds: Option<i64>,
    pub partition: EvaluationPartition,
}

/// Requirements fixed before split integrity is audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitIntegrityPolicy {
    pub minimum_clusters_per_partition: usize,
    /// If enabled, each partition's latest event must precede the next one's earliest.
    /// Disable for grouped/OOD holdouts that are not intended to be chronological.
    pub require_strict_chronological_order: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitAuditStatus {
    Accept,
    Reject,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitAuditFinding {
    EmptyRecordSet,
    InvalidMinimumClusterCount,
    EmptyRecordId,
    EmptyEventClusterId,
    DuplicateRecordId { record_id: String },
    MissingEventTime { record_id: String },
    EventClusterCrossesPartitions {
        event_cluster_id: String,
        first_partition: EvaluationPartition,
        second_partition: EvaluationPartition,
    },
    InsufficientClusters {
        partition: EvaluationPartition,
        observed: usize,
        minimum: usize,
    },
    NonChronologicalBoundary {
        earlier_partition: EvaluationPartition,
        later_partition: EvaluationPartition,
        earlier_max_epoch_seconds: i64,
        later_min_epoch_seconds: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionStats {
    pub partition: EvaluationPartition,
    pub row_count: usize,
    pub distinct_event_clusters: usize,
    pub earliest_epoch_seconds: Option<i64>,
    pub latest_epoch_seconds: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitIntegrityReport {
    pub status: SplitAuditStatus,
    pub findings: Vec<SplitAuditFinding>,
    pub partitions: Vec<PartitionStats>,
}

/// Audit split metadata before calibrating a radius or reporting final results.
///
/// A cluster may contain multiple rows, but a cluster may occur in only one
/// partition. Row IDs must be unique. If chronology is enabled, every event time
/// must be present and partition time intervals must be strictly separated.
pub fn audit_split_integrity(
    records: &[SplitRecord],
    policy: SplitIntegrityPolicy,
) -> SplitIntegrityReport {
    let mut findings = Vec::new();
    let mut hard_reject = false;
    let mut indeterminate = false;

    if records.is_empty() {
        findings.push(SplitAuditFinding::EmptyRecordSet);
        hard_reject = true;
    }
    if policy.minimum_clusters_per_partition == 0 {
        findings.push(SplitAuditFinding::InvalidMinimumClusterCount);
        hard_reject = true;
    }

    let mut seen_records = BTreeSet::new();
    let mut cluster_partitions: BTreeMap<String, EvaluationPartition> = BTreeMap::new();
    let mut row_counts: BTreeMap<EvaluationPartition, usize> = BTreeMap::new();
    let mut cluster_sets: BTreeMap<EvaluationPartition, BTreeSet<String>> = BTreeMap::new();
    let mut min_time: BTreeMap<EvaluationPartition, i64> = BTreeMap::new();
    let mut max_time: BTreeMap<EvaluationPartition, i64> = BTreeMap::new();

    for record in records {
        if record.record_id.trim().is_empty() {
            findings.push(SplitAuditFinding::EmptyRecordId);
            hard_reject = true;
        } else if !seen_records.insert(record.record_id.clone()) {
            findings.push(SplitAuditFinding::DuplicateRecordId {
                record_id: record.record_id.clone(),
            });
            hard_reject = true;
        }

        if record.event_cluster_id.trim().is_empty() {
            findings.push(SplitAuditFinding::EmptyEventClusterId);
            hard_reject = true;
        } else {
            cluster_sets
                .entry(record.partition)
                .or_default()
                .insert(record.event_cluster_id.clone());
            match cluster_partitions.get(&record.event_cluster_id) {
                Some(previous) if *previous != record.partition => {
                    findings.push(SplitAuditFinding::EventClusterCrossesPartitions {
                        event_cluster_id: record.event_cluster_id.clone(),
                        first_partition: *previous,
                        second_partition: record.partition,
                    });
                    hard_reject = true;
                }
                None => {
                    cluster_partitions.insert(record.event_cluster_id.clone(), record.partition);
                }
                _ => {}
            }
        }

        *row_counts.entry(record.partition).or_default() += 1;
        match record.event_time_epoch_seconds {
            Some(time) => {
                min_time
                    .entry(record.partition)
                    .and_modify(|current| *current = (*current).min(time))
                    .or_insert(time);
                max_time
                    .entry(record.partition)
                    .and_modify(|current| *current = (*current).max(time))
                    .or_insert(time);
            }
            None => {
                findings.push(SplitAuditFinding::MissingEventTime {
                    record_id: record.record_id.clone(),
                });
                indeterminate = true;
            }
        }
    }

    let mut partitions = Vec::with_capacity(4);
    for partition in EvaluationPartition::ordered() {
        let clusters = cluster_sets.get(&partition).map_or(0, BTreeSet::len);
        let stats = PartitionStats {
            partition,
            row_count: row_counts.get(&partition).copied().unwrap_or(0),
            distinct_event_clusters: clusters,
            earliest_epoch_seconds: min_time.get(&partition).copied(),
            latest_epoch_seconds: max_time.get(&partition).copied(),
        };
        if policy.minimum_clusters_per_partition > 0
            && clusters < policy.minimum_clusters_per_partition
        {
            findings.push(SplitAuditFinding::InsufficientClusters {
                partition,
                observed: clusters,
                minimum: policy.minimum_clusters_per_partition,
            });
            indeterminate = true;
        }
        partitions.push(stats);
    }

    if policy.require_strict_chronological_order {
        for pair in EvaluationPartition::ordered().windows(2) {
            let earlier = pair[0];
            let later = pair[1];
            if let (Some(earlier_max), Some(later_min)) =
                (max_time.get(&earlier), min_time.get(&later))
            {
                if earlier_max >= later_min {
                    findings.push(SplitAuditFinding::NonChronologicalBoundary {
                        earlier_partition: earlier,
                        later_partition: later,
                        earlier_max_epoch_seconds: *earlier_max,
                        later_min_epoch_seconds: *later_min,
                    });
                    hard_reject = true;
                }
            }
        }
    }

    let status = if hard_reject {
        SplitAuditStatus::Reject
    } else if indeterminate {
        SplitAuditStatus::Indeterminate
    } else {
        SplitAuditStatus::Accept
    };

    SplitIntegrityReport { status, findings, partitions }
}

/// One declared benchmark choice whose parameters may be tuned from a development split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningChoice {
    /// Stable declared artifact, such as tv_radius, risk_threshold or policy_id.
    pub artifact_id: String,
    pub selected_using: EvaluationPartition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningAuditReport {
    pub status: SplitAuditStatus,
    pub offending_artifacts: Vec<String>,
}

/// Reject tuning declarations that use the final test partition or are ambiguous.
///
/// This is a process-contract check over declared provenance, not a guarantee
/// that analysts or external tools did not inspect test outcomes.
pub fn audit_no_test_set_tuning(choices: &[TuningChoice]) -> TuningAuditReport {
    if choices.is_empty() {
        return TuningAuditReport {
            status: SplitAuditStatus::Indeterminate,
            offending_artifacts: Vec::new(),
        };
    }

    let mut seen = BTreeSet::new();
    let mut offending = BTreeSet::new();
    let mut hard_reject = false;
    for choice in choices {
        if choice.artifact_id.trim().is_empty() || !seen.insert(choice.artifact_id.clone()) {
            offending.insert(choice.artifact_id.clone());
            hard_reject = true;
        }
        if choice.selected_using == EvaluationPartition::Test {
            offending.insert(choice.artifact_id.clone());
            hard_reject = true;
        }
    }

    TuningAuditReport {
        status: if hard_reject { SplitAuditStatus::Reject } else { SplitAuditStatus::Accept },
        offending_artifacts: offending.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<SplitRecord> {
        let partitions = [
            (EvaluationPartition::Training, 100),
            (EvaluationPartition::Calibration, 200),
            (EvaluationPartition::Validation, 300),
            (EvaluationPartition::Test, 400),
        ];
        let mut rows = Vec::new();
        for (partition, timestamp) in partitions {
            for index in 0..3 {
                rows.push(SplitRecord {
                    record_id: format!("{partition:?}-{index}"),
                    event_cluster_id: format!("{partition:?}-cluster-{index}"),
                    event_time_epoch_seconds: Some(timestamp + index),
                    partition,
                });
            }
        }
        rows
    }

    fn policy() -> SplitIntegrityPolicy {
        SplitIntegrityPolicy {
            minimum_clusters_per_partition: 3,
            require_strict_chronological_order: true,
        }
    }

    #[test]
    fn accepts_disjoint_sufficient_chronological_splits() {
        let report = audit_split_integrity(&fixture(), policy());
        assert_eq!(report.status, SplitAuditStatus::Accept);
        assert!(report.findings.is_empty());
        assert_eq!(report.partitions.len(), 4);
    }

    #[test]
    fn rejects_event_cluster_leakage_even_when_rows_differ() {
        let mut rows = fixture();
        rows[9].event_cluster_id = rows[0].event_cluster_id.clone();
        let report = audit_split_integrity(&rows, policy());
        assert_eq!(report.status, SplitAuditStatus::Reject);
        assert!(report.findings.iter().any(|f| matches!(f, SplitAuditFinding::EventClusterCrossesPartitions { .. })));
    }

    #[test]
    fn rejects_duplicate_record_ids() {
        let mut rows = fixture();
        rows[1].record_id = rows[0].record_id.clone();
        let report = audit_split_integrity(&rows, policy());
        assert_eq!(report.status, SplitAuditStatus::Reject);
        assert!(report.findings.iter().any(|f| matches!(f, SplitAuditFinding::DuplicateRecordId { .. })));
    }

    #[test]
    fn rejects_non_chronological_partition_boundary() {
        let mut rows = fixture();
        rows[9].event_time_epoch_seconds = Some(150);
        let report = audit_split_integrity(&rows, policy());
        assert_eq!(report.status, SplitAuditStatus::Reject);
        assert!(report.findings.iter().any(|f| matches!(f, SplitAuditFinding::NonChronologicalBoundary { .. })));
    }

    #[test]
    fn missing_timestamps_make_audit_indeterminate() {
        let mut rows = fixture();
        rows[0].event_time_epoch_seconds = None;
        let report = audit_split_integrity(&rows, policy());
        assert_eq!(report.status, SplitAuditStatus::Indeterminate);
        assert!(report.findings.iter().any(|f| matches!(f, SplitAuditFinding::MissingEventTime { .. })));
    }

    #[test]
    fn insufficient_clusters_are_indeterminate_not_silently_accepted() {
        let mut rows = fixture();
        rows.retain(|row| !(row.partition == EvaluationPartition::Test && row.event_cluster_id.ends_with("cluster-2")));
        let report = audit_split_integrity(&rows, policy());
        assert_eq!(report.status, SplitAuditStatus::Indeterminate);
    }

    #[test]
    fn rejects_zero_minimum_cluster_requirement() {
        let report = audit_split_integrity(
            &fixture(),
            SplitIntegrityPolicy {
                minimum_clusters_per_partition: 0,
                require_strict_chronological_order: false,
            },
        );
        assert_eq!(report.status, SplitAuditStatus::Reject);
    }

    #[test]
    fn chronology_can_be_disabled_for_grouped_ood_holdouts() {
        let mut rows = fixture();
        rows[9].event_time_epoch_seconds = Some(101);
        let report = audit_split_integrity(
            &rows,
            SplitIntegrityPolicy {
                minimum_clusters_per_partition: 3,
                require_strict_chronological_order: false,
            },
        );
        assert_eq!(report.status, SplitAuditStatus::Accept);
    }

    #[test]
    fn tuning_audit_rejects_radius_chosen_from_test_results() {
        let report = audit_no_test_set_tuning(&[TuningChoice {
            artifact_id: "tv_radius".into(),
            selected_using: EvaluationPartition::Test,
        }]);
        assert_eq!(report.status, SplitAuditStatus::Reject);
        assert_eq!(report.offending_artifacts, vec!["tv_radius"]);
    }

    #[test]
    fn tuning_audit_accepts_development_only_choices() {
        let report = audit_no_test_set_tuning(&[
            TuningChoice { artifact_id: "tv_radius".into(), selected_using: EvaluationPartition::Calibration },
            TuningChoice { artifact_id: "risk_threshold".into(), selected_using: EvaluationPartition::Validation },
        ]);
        assert_eq!(report.status, SplitAuditStatus::Accept);
    }

    #[test]
    fn tuning_audit_rejects_duplicate_or_empty_artifact_ids() {
        let report = audit_no_test_set_tuning(&[
            TuningChoice { artifact_id: "radius".into(), selected_using: EvaluationPartition::Calibration },
            TuningChoice { artifact_id: "radius".into(), selected_using: EvaluationPartition::Validation },
        ]);
        assert_eq!(report.status, SplitAuditStatus::Reject);
        assert_eq!(report.offending_artifacts, vec!["radius"]);
    }

    #[test]
    fn empty_tuning_manifest_is_indeterminate() {
        assert_eq!(audit_no_test_set_tuning(&[]).status, SplitAuditStatus::Indeterminate);
    }

    #[test]
    fn row_order_does_not_change_partition_stats() {
        let rows = fixture();
        let report_a = audit_split_integrity(&rows, policy());
        let mut reversed = rows;
        reversed.reverse();
        let report_b = audit_split_integrity(&reversed, policy());
        assert_eq!(report_a.status, report_b.status);
        assert_eq!(report_a.partitions, report_b.partitions);
    }

    #[test]
    fn both_missing_evidence_and_small_partition_remain_indeterminate() {
        let mut rows = fixture();
        rows[0].event_time_epoch_seconds = None;
        rows.retain(|row| !(row.partition == EvaluationPartition::Test && row.event_cluster_id.ends_with("cluster-2")));
        let report = audit_split_integrity(&rows, policy());
        assert_eq!(report.status, SplitAuditStatus::Indeterminate);
        assert_eq!(report.findings.iter().filter(|f| matches!(f, SplitAuditFinding::InsufficientClusters { .. })).count(), 1);
    }
}
