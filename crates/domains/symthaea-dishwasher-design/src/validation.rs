//! Leakage checks for physically grouped dishwasher validation datasets.
//!
//! One record represents one physical wash run. Item observations, images,
//! sensor frames and assay replicates must reference that run and inherit its
//! partition; they are not separate independent trials.
//!
//! This module checks declared group separation only. It cannot authenticate
//! identifiers, establish physical independence, estimate cluster correlation,
//! validate an assay or make a performance/certification claim.

use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValidationPartition {
    Development,
    Calibration,
    LockedConfirmation,
    ExternalReplication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HoldoutAxis {
    MachineUnit,
    TestSession,
    SoilBatch,
    DetergentBatch,
    WaterBatch,
    Operator,
    AssayLab,
    TestSite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunPartitionRecord {
    /// Unique ID for one complete physical wash run/cycle.
    pub run_id: String,
    pub partition: ValidationPartition,
    pub machine_unit_id: String,
    pub test_session_id: String,
    pub soil_batch_id: String,
    pub detergent_batch_id: String,
    pub water_batch_id: String,
    pub operator_id: String,
    pub assay_lab_id: String,
    pub test_site_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupedValidationError {
    EmptyRuns,
    EmptyHoldoutAxes,
    EmptyRequiredPartitions,
    DuplicateHoldoutAxis(HoldoutAxis),
    DuplicateRequiredPartition(ValidationPartition),
    InvalidIdentity(&'static str),
    DuplicateRunId(String),
    EmptyDerivedRecords,
    DuplicateDerivedRecordId(String),
    UnknownSourceRun(String),
    DerivedPartitionMismatch {
        record_id: String,
        source_run_id: String,
        expected: ValidationPartition,
        observed: ValidationPartition,
    },
    GroupLeaksAcrossPartitions {
        axis: HoldoutAxis,
        group_id: String,
        first_partition: ValidationPartition,
        second_partition: ValidationPartition,
    },
    MissingRequiredPartition(ValidationPartition),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedValidationReport {
    pub physical_run_count: usize,
    /// Counts are returned in enum order for deterministic serialization.
    pub partition_counts: Vec<(ValidationPartition, usize)>,
    /// Number of distinct held-out groups, returned in requested-axis order.
    pub held_out_group_counts: Vec<(HoldoutAxis, usize)>,
}

fn canonical_non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

impl HoldoutAxis {
    fn group_id<'a>(self, record: &'a RunPartitionRecord) -> &'a str {
        match self {
            HoldoutAxis::MachineUnit => &record.machine_unit_id,
            HoldoutAxis::TestSession => &record.test_session_id,
            HoldoutAxis::SoilBatch => &record.soil_batch_id,
            HoldoutAxis::DetergentBatch => &record.detergent_batch_id,
            HoldoutAxis::WaterBatch => &record.water_batch_id,
            HoldoutAxis::Operator => &record.operator_id,
            HoldoutAxis::AssayLab => &record.assay_lab_id,
            HoldoutAxis::TestSite => &record.test_site_id,
        }
    }
}

/// Reject any declared holdout group that occurs in more than one partition.
///
/// `holdout_axes` is explicit: a new-day test may intentionally reuse a
/// machine unit, while a cross-unit test must include `MachineUnit`. A combined
/// external-shift test can request several axes at once.
///
/// Each physical run ID must occur exactly once in this manifest. All derived
/// item/image/frame/assay rows must join to this record by run ID and inherit
/// its partition outside this function. Hashes and IDs are opaque identifiers,
/// not authentication or proof of independence.
pub fn audit_grouped_holdout(
    records: &[RunPartitionRecord],
    holdout_axes: &[HoldoutAxis],
    required_partitions: &[ValidationPartition],
) -> Result<GroupedValidationReport, GroupedValidationError> {
    if records.is_empty() {
        return Err(GroupedValidationError::EmptyRuns);
    }
    if holdout_axes.is_empty() {
        return Err(GroupedValidationError::EmptyHoldoutAxes);
    }
    if required_partitions.is_empty() {
        return Err(GroupedValidationError::EmptyRequiredPartitions);
    }

    let mut axes = BTreeSet::new();
    for axis in holdout_axes {
        if !axes.insert(*axis) {
            return Err(GroupedValidationError::DuplicateHoldoutAxis(*axis));
        }
    }

    let mut required = BTreeSet::new();
    for partition in required_partitions {
        if !required.insert(*partition) {
            return Err(GroupedValidationError::DuplicateRequiredPartition(*partition));
        }
    }

    let mut run_ids = HashSet::with_capacity(records.len());
    let mut group_owners: BTreeMap<(HoldoutAxis, String), ValidationPartition> =
        BTreeMap::new();
    let mut group_values: BTreeMap<HoldoutAxis, BTreeSet<String>> = BTreeMap::new();
    let mut counts: BTreeMap<ValidationPartition, usize> = BTreeMap::new();

    for record in records {
        if !canonical_non_empty(&record.run_id) {
            return Err(GroupedValidationError::InvalidIdentity("run_id"));
        }
        for (field, value) in [
            ("machine_unit_id", record.machine_unit_id.as_str()),
            ("test_session_id", record.test_session_id.as_str()),
            ("soil_batch_id", record.soil_batch_id.as_str()),
            ("detergent_batch_id", record.detergent_batch_id.as_str()),
            ("water_batch_id", record.water_batch_id.as_str()),
            ("operator_id", record.operator_id.as_str()),
            ("assay_lab_id", record.assay_lab_id.as_str()),
            ("test_site_id", record.test_site_id.as_str()),
        ] {
            if !canonical_non_empty(value) {
                return Err(GroupedValidationError::InvalidIdentity(field));
            }
        }
        if !run_ids.insert(record.run_id.as_str()) {
            return Err(GroupedValidationError::DuplicateRunId(
                record.run_id.clone(),
            ));
        }

        *counts.entry(record.partition).or_insert(0) += 1;

        for axis in holdout_axes {
            let group_id = axis.group_id(record);
            let key = (*axis, group_id.to_owned());
            match group_owners.get(&key) {
                Some(first_partition) if *first_partition != record.partition => {
                    return Err(GroupedValidationError::GroupLeaksAcrossPartitions {
                        axis: *axis,
                        group_id: group_id.to_owned(),
                        first_partition: *first_partition,
                        second_partition: record.partition,
                    });
                }
                None => {
                    group_owners.insert(key, record.partition);
                }
                _ => {}
            }
            group_values
                .entry(*axis)
                .or_default()
                .insert(group_id.to_owned());
        }
    }

    for partition in required_partitions {
        if counts.get(partition).copied().unwrap_or(0) == 0 {
            return Err(GroupedValidationError::MissingRequiredPartition(*partition));
        }
    }

    let partition_counts = [
        ValidationPartition::Development,
        ValidationPartition::Calibration,
        ValidationPartition::LockedConfirmation,
        ValidationPartition::ExternalReplication,
    ]
    .into_iter()
    .filter_map(|partition| {
        counts
            .get(&partition)
            .copied()
            .map(|count| (partition, count))
    })
    .collect();

    let held_out_group_counts = holdout_axes
        .iter()
        .map(|axis| {
            (
                *axis,
                group_values.get(axis).map_or(0, BTreeSet::len),
            )
        })
        .collect();

    Ok(GroupedValidationReport {
        physical_run_count: records.len(),
        partition_counts,
        held_out_group_counts,
    })
}


/// One item/image/frame/assay record linked to its source physical run.
///
/// Multiple derived records may reference the same source run, but they must
/// all inherit the partition assigned to that run in the physical-run manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedRecordPartition {
    pub record_id: String,
    pub source_run_id: String,
    pub partition: ValidationPartition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedPartitionReport {
    pub source_run_count: usize,
    pub derived_record_count: usize,
    pub unique_source_run_count: usize,
}

/// Verify that every derived record references a known physical run and
/// inherits exactly that run's partition.
///
/// This check prevents one run's images, item observations or assay outputs
/// from being manually re-assigned to a different split. It still depends on
/// trustworthy upstream run IDs and does not authenticate the underlying data.
pub fn audit_derived_record_partitions(
    runs: &[RunPartitionRecord],
    derived_records: &[DerivedRecordPartition],
) -> Result<DerivedPartitionReport, GroupedValidationError> {
    if runs.is_empty() {
        return Err(GroupedValidationError::EmptyRuns);
    }
    if derived_records.is_empty() {
        return Err(GroupedValidationError::EmptyDerivedRecords);
    }

    let mut partition_by_run: BTreeMap<&str, ValidationPartition> = BTreeMap::new();
    for run in runs {
        if !canonical_non_empty(&run.run_id) {
            return Err(GroupedValidationError::InvalidIdentity("run_id"));
        }
        if partition_by_run.insert(run.run_id.as_str(), run.partition).is_some() {
            return Err(GroupedValidationError::DuplicateRunId(run.run_id.clone()));
        }
    }

    let mut derived_ids = HashSet::with_capacity(derived_records.len());
    let mut linked_run_ids = HashSet::with_capacity(derived_records.len());
    for derived in derived_records {
        if !canonical_non_empty(&derived.record_id) {
            return Err(GroupedValidationError::InvalidIdentity("derived_record_id"));
        }
        if !canonical_non_empty(&derived.source_run_id) {
            return Err(GroupedValidationError::InvalidIdentity("source_run_id"));
        }
        if !derived_ids.insert(derived.record_id.as_str()) {
            return Err(GroupedValidationError::DuplicateDerivedRecordId(
                derived.record_id.clone(),
            ));
        }
        let expected = partition_by_run
            .get(derived.source_run_id.as_str())
            .copied()
            .ok_or_else(|| {
                GroupedValidationError::UnknownSourceRun(derived.source_run_id.clone())
            })?;
        if expected != derived.partition {
            return Err(GroupedValidationError::DerivedPartitionMismatch {
                record_id: derived.record_id.clone(),
                source_run_id: derived.source_run_id.clone(),
                expected,
                observed: derived.partition,
            });
        }
        linked_run_ids.insert(derived.source_run_id.as_str());
    }

    Ok(DerivedPartitionReport {
        source_run_count: runs.len(),
        derived_record_count: derived_records.len(),
        unique_source_run_count: linked_run_ids.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        run_id: &str,
        partition: ValidationPartition,
        machine: &str,
        session: &str,
        soil_batch: &str,
    ) -> RunPartitionRecord {
        RunPartitionRecord {
            run_id: run_id.to_owned(),
            partition,
            machine_unit_id: machine.to_owned(),
            test_session_id: session.to_owned(),
            soil_batch_id: soil_batch.to_owned(),
            detergent_batch_id: format!("detergent-{session}"),
            water_batch_id: format!("water-{session}"),
            operator_id: format!("operator-{session}"),
            assay_lab_id: format!("lab-{session}"),
            test_site_id: format!("site-{session}"),
        }
    }

    #[test]
    fn accepts_disjoint_machine_and_session_groups() {
        let records = [
            record("run-dev", ValidationPartition::Development, "unit-a", "day-a", "soil-a"),
            record("run-cal", ValidationPartition::Calibration, "unit-b", "day-b", "soil-b"),
        ];
        let report = audit_grouped_holdout(
            &records,
            &[HoldoutAxis::MachineUnit, HoldoutAxis::TestSession],
            &[
                ValidationPartition::Development,
                ValidationPartition::Calibration,
            ],
        )
        .unwrap();
        assert_eq!(report.physical_run_count, 2);
        assert_eq!(report.partition_counts.len(), 2);
        assert_eq!(report.held_out_group_counts.len(), 2);
    }

    #[test]
    fn rejects_machine_unit_leakage_across_partitions() {
        let records = [
            record("run-dev", ValidationPartition::Development, "same-unit", "day-a", "soil-a"),
            record("run-test", ValidationPartition::LockedConfirmation, "same-unit", "day-b", "soil-b"),
        ];
        let result = audit_grouped_holdout(
            &records,
            &[HoldoutAxis::MachineUnit],
            &[
                ValidationPartition::Development,
                ValidationPartition::LockedConfirmation,
            ],
        );
        assert!(matches!(
            result,
            Err(GroupedValidationError::GroupLeaksAcrossPartitions {
                axis: HoldoutAxis::MachineUnit,
                ..
            })
        ));
    }

    #[test]
    fn allows_reused_machine_when_only_new_day_is_the_declared_holdout() {
        let records = [
            record("run-dev", ValidationPartition::Development, "same-unit", "day-a", "soil-a"),
            record("run-test", ValidationPartition::LockedConfirmation, "same-unit", "day-b", "soil-a"),
        ];
        assert!(audit_grouped_holdout(
            &records,
            &[HoldoutAxis::TestSession],
            &[
                ValidationPartition::Development,
                ValidationPartition::LockedConfirmation,
            ],
        )
        .is_ok());
    }

    #[test]
    fn rejects_duplicate_physical_run_ids() {
        let records = [
            record("same-run", ValidationPartition::Development, "unit-a", "day-a", "soil-a"),
            record("same-run", ValidationPartition::Calibration, "unit-b", "day-b", "soil-b"),
        ];
        assert_eq!(
            audit_grouped_holdout(
                &records,
                &[HoldoutAxis::TestSession],
                &[
                    ValidationPartition::Development,
                    ValidationPartition::Calibration,
                ],
            ),
            Err(GroupedValidationError::DuplicateRunId("same-run".to_owned()))
        );
    }

    #[test]
    fn rejects_missing_group_identity() {
        let mut item = record(
            "run-a",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        );
        item.water_batch_id.clear();
        assert_eq!(
            audit_grouped_holdout(
                &[item],
                &[HoldoutAxis::TestSession],
                &[ValidationPartition::Development],
            ),
            Err(GroupedValidationError::InvalidIdentity("water_batch_id"))
        );
    }

    #[test]
    fn requires_each_declared_partition_to_be_present() {
        let records = [record(
            "run-a",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        assert_eq!(
            audit_grouped_holdout(
                &records,
                &[HoldoutAxis::TestSession],
                &[
                    ValidationPartition::Development,
                    ValidationPartition::LockedConfirmation,
                ],
            ),
            Err(GroupedValidationError::MissingRequiredPartition(
                ValidationPartition::LockedConfirmation
            ))
        );
    }

    #[test]
    fn rejects_empty_or_duplicate_holdout_axes() {
        let records = [record(
            "run-a",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        assert_eq!(
            audit_grouped_holdout(
                &records,
                &[],
                &[ValidationPartition::Development],
            ),
            Err(GroupedValidationError::EmptyHoldoutAxes)
        );
        assert_eq!(
            audit_grouped_holdout(
                &records,
                &[HoldoutAxis::TestSession, HoldoutAxis::TestSession],
                &[ValidationPartition::Development],
            ),
            Err(GroupedValidationError::DuplicateHoldoutAxis(
                HoldoutAxis::TestSession
            ))
        );
    }

    #[test]
    fn rejects_empty_required_partitions() {
        let records = [record(
            "run-a",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        assert_eq!(
            audit_grouped_holdout(&records, &[HoldoutAxis::TestSession], &[]),
            Err(GroupedValidationError::EmptyRequiredPartitions)
        );
    }

    #[test]
    fn combined_external_shift_requires_every_declared_axis_to_be_disjoint() {
        let records = [
            record("run-dev", ValidationPartition::Development, "unit-a", "day-a", "soil-a"),
            record("run-external", ValidationPartition::ExternalReplication, "unit-b", "day-b", "soil-b"),
        ];
        assert!(audit_grouped_holdout(
            &records,
            &[
                HoldoutAxis::MachineUnit,
                HoldoutAxis::TestSession,
                HoldoutAxis::SoilBatch,
                HoldoutAxis::DetergentBatch,
                HoldoutAxis::WaterBatch,
                HoldoutAxis::Operator,
                HoldoutAxis::AssayLab,
                HoldoutAxis::TestSite,
            ],
            &[
                ValidationPartition::Development,
                ValidationPartition::ExternalReplication,
            ],
        )
        .is_ok());
    }

    #[test]
    fn derived_records_inherit_run_partition_and_allow_multiple_records_per_run() {
        let runs = [
            record("run-dev", ValidationPartition::Development, "unit-a", "day-a", "soil-a"),
            record("run-test", ValidationPartition::LockedConfirmation, "unit-b", "day-b", "soil-b"),
        ];
        let derived = [
            DerivedRecordPartition {
                record_id: "image-1".to_owned(),
                source_run_id: "run-dev".to_owned(),
                partition: ValidationPartition::Development,
            },
            DerivedRecordPartition {
                record_id: "assay-1".to_owned(),
                source_run_id: "run-dev".to_owned(),
                partition: ValidationPartition::Development,
            },
            DerivedRecordPartition {
                record_id: "image-2".to_owned(),
                source_run_id: "run-test".to_owned(),
                partition: ValidationPartition::LockedConfirmation,
            },
        ];
        let report = audit_derived_record_partitions(&runs, &derived).unwrap();
        assert_eq!(report.source_run_count, 2);
        assert_eq!(report.derived_record_count, 3);
        assert_eq!(report.unique_source_run_count, 2);
    }

    #[test]
    fn rejects_derived_record_link_to_unknown_run() {
        let runs = [record(
            "run-dev",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        let derived = [DerivedRecordPartition {
            record_id: "image-x".to_owned(),
            source_run_id: "missing-run".to_owned(),
            partition: ValidationPartition::Development,
        }];
        assert_eq!(
            audit_derived_record_partitions(&runs, &derived),
            Err(GroupedValidationError::UnknownSourceRun(
                "missing-run".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_derived_record_partition_that_differs_from_source_run() {
        let runs = [record(
            "run-dev",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        let derived = [DerivedRecordPartition {
            record_id: "image-x".to_owned(),
            source_run_id: "run-dev".to_owned(),
            partition: ValidationPartition::LockedConfirmation,
        }];
        assert!(matches!(
            audit_derived_record_partitions(&runs, &derived),
            Err(GroupedValidationError::DerivedPartitionMismatch {
                expected: ValidationPartition::Development,
                observed: ValidationPartition::LockedConfirmation,
                ..
            })
        ));
    }

    #[test]
    fn rejects_duplicate_derived_record_ids() {
        let runs = [record(
            "run-dev",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        let derived = [
            DerivedRecordPartition {
                record_id: "duplicate".to_owned(),
                source_run_id: "run-dev".to_owned(),
                partition: ValidationPartition::Development,
            },
            DerivedRecordPartition {
                record_id: "duplicate".to_owned(),
                source_run_id: "run-dev".to_owned(),
                partition: ValidationPartition::Development,
            },
        ];
        assert_eq!(
            audit_derived_record_partitions(&runs, &derived),
            Err(GroupedValidationError::DuplicateDerivedRecordId(
                "duplicate".to_owned()
            ))
        );
    }

    #[test]
    fn rejects_empty_derived_record_collection() {
        let runs = [record(
            "run-dev",
            ValidationPartition::Development,
            "unit-a",
            "day-a",
            "soil-a",
        )];
        assert_eq!(
            audit_derived_record_partitions(&runs, &[]),
            Err(GroupedValidationError::EmptyDerivedRecords)
        );
    }

}
