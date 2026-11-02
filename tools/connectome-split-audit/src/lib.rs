#![forbid(unsafe_code)]

//! Group-disjoint split and task-view leakage auditor for connectomic proofreading datasets.
//!
//! The audit unit is a task-routed operation derived from a stable sample hash. Multiple
//! tasks may derive from one sample, but they must inherit the same split. Every image/view
//! derivative inherits the parent operation's split and task route.
//!
//! This crate checks manifest consistency and detects group overlap. It does not prove that
//! source group identifiers are biologically correct or exhaustive; the data adapter must
//! populate them from pinned source metadata and preserve missing values explicitly.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSplit {
    Train,
    Validation,
    Test,
    Adaptation,
    Calibration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskRoute {
    FalseSplitCorrection,
    FalseMergeIdentification,
    SplitMaskGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GroupKey {
    SourceArchive,
    SourceVolume,
    SourceOperation,
    SpatialCube,
    Neuron,
    Animal,
    Session,
    Species,
    Workflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViewKind {
    EmXy,
    EmXz,
    EmYz,
    EmBest,
    GeometrySingle,
    GeometryDual,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupIdentity {
    pub source_archive_id: Option<String>,
    pub source_volume_id: Option<String>,
    /// Source-qualified upstream source_operation_id from original proofreading metadata.
    pub source_operation_id: Option<String>,
    /// Source-volume-qualified 50-micrometre cube identity from the pinned split manifest.
    pub spatial_cube_id: Option<String>,
    /// Union of before/after neuron root IDs, canonicalized by the data adapter.
    pub neuron_ids: Vec<String>,
    pub animal_id: Option<String>,
    pub session_id: Option<String>,
    pub species: Option<String>,
    pub workflow_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewInstance {
    pub view_id: String,
    pub sample_hash: String,
    pub split: DataSplit,
    pub task_route: TaskRoute,
    pub view_kind: ViewKind,
    pub artifact_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationRecord {
    /// Stable operation identity (normally sample_hash + task route).
    pub operation_id: String,
    /// ConnectomeBench2 combined_sample_hash: 32 lowercase hexadecimal characters.
    pub sample_hash: String,
    pub split: DataSplit,
    pub task_route: TaskRoute,
    pub groups: GroupIdentity,
    pub label_artifact_sha256: String,
    pub views: Vec<ViewInstance>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ViewPermission {
    pub task_route: TaskRoute,
    pub view_kind: ViewKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitAuditPolicy {
    pub policy_id: String,
    /// Each listed identity must be present and disjoint across splits.
    pub group_keys: Vec<GroupKey>,
    /// Explicitly permitted view kinds for each task.
    pub view_permissions: Vec<ViewPermission>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupOverlap {
    pub group_key: GroupKey,
    pub group_value: String,
    pub splits: Vec<DataSplit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitSummary {
    pub split: DataSplit,
    pub operation_count: usize,
    pub unique_sample_count: usize,
    pub view_instance_count: usize,
    pub task_counts: BTreeMap<TaskRoute, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitAuditStatus {
    GroupDisjoint,
    LeakageDetected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitAuditReport {
    pub report_version: &'static str,
    pub policy_id: String,
    pub operation_count: usize,
    pub unique_sample_count: usize,
    pub view_instance_count: usize,
    pub group_keys_audited: Vec<GroupKey>,
    pub split_summaries: Vec<SplitSummary>,
    pub group_overlaps: Vec<GroupOverlap>,
    pub status: SplitAuditStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitAuditError {
    InvalidIdentifier { kind: &'static str, value: String },
    InvalidSampleHash(String),
    InvalidSha256 { kind: &'static str, value: String },
    EmptyCollection(&'static str),
    DuplicateIdentifier { kind: &'static str, value: String },
    DuplicateOperationForTask { sample_hash: String, task_route: TaskRoute },
    SampleAppearsInMultipleSplits {
        sample_hash: String,
        first: DataSplit,
        second: DataSplit,
    },
    SampleGroupIdentityMismatch { sample_hash: String },
    MissingGroupKey { operation_id: String, group_key: GroupKey },
    MissingViewPermission { task_route: TaskRoute, view_kind: ViewKind },
    ViewSampleMismatch { operation_id: String, view_id: String },
    ViewSplitMismatch { operation_id: String, view_id: String },
    ViewTaskMismatch { operation_id: String, view_id: String },
    TargetLeakingView { operation_id: String, view_id: String },
}

impl fmt::Display for SplitAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::InvalidSampleHash(hash) => {
                write!(f, "sample hash must be exactly 32 lowercase hex characters: {hash:?}")
            }
            Self::InvalidSha256 { kind, value } => {
                write!(f, "invalid SHA-256 for {kind}: {value:?}")
            }
            Self::EmptyCollection(name) => write!(f, "{name} must not be empty"),
            Self::DuplicateIdentifier { kind, value } => {
                write!(f, "duplicate {kind} identifier: {value}")
            }
            Self::DuplicateOperationForTask { sample_hash, task_route } => {
                write!(f, "duplicate task-routed operation for sample {sample_hash}: {task_route:?}")
            }
            Self::SampleAppearsInMultipleSplits {
                sample_hash,
                first,
                second,
            } => write!(
                f,
                "sample {sample_hash} appears in multiple splits: {first:?} and {second:?}"
            ),
            Self::SampleGroupIdentityMismatch { sample_hash } => write!(
                f,
                "task routes for sample {sample_hash} disagree on source group identity"
            ),
            Self::MissingGroupKey { operation_id, group_key } => {
                write!(f, "operation {operation_id} is missing group key {group_key:?}")
            }
            Self::MissingViewPermission { task_route, view_kind } => {
                write!(f, "view kind {view_kind:?} is not permitted for task {task_route:?}")
            }
            Self::ViewSampleMismatch { operation_id, view_id } => {
                write!(f, "view {view_id} does not inherit sample identity from {operation_id}")
            }
            Self::ViewSplitMismatch { operation_id, view_id } => {
                write!(f, "view {view_id} does not inherit split from {operation_id}")
            }
            Self::ViewTaskMismatch { operation_id, view_id } => {
                write!(f, "view {view_id} does not inherit task route from {operation_id}")
            }
            Self::TargetLeakingView { operation_id, view_id } => {
                write!(f, "target-leaking view {view_id} is forbidden for operation {operation_id}")
            }
        }
    }
}

impl std::error::Error for SplitAuditError {}

/// Audit one frozen operation/view manifest under an explicit group-disjoint split policy.
///
/// Duplicate row/sample/task identities and malformed view ancestry fail closed. Legitimate
/// overlap of a declared group key across splits is returned as LeakageDetected with the
/// exact overlapping values and split names; consumers must not treat that report as clean.
pub fn audit_splits(
    policy: &SplitAuditPolicy,
    operations: &[OperationRecord],
) -> Result<SplitAuditReport, SplitAuditError> {
    validate_policy(policy)?;
    if operations.is_empty() {
        return Err(SplitAuditError::EmptyCollection("operations"));
    }

    let mut operation_ids = BTreeSet::new();
    let mut view_ids = BTreeSet::new();
    let mut sample_task_pairs = BTreeSet::new();
    let mut sample_splits: BTreeMap<String, DataSplit> = BTreeMap::new();
    let mut sample_group_identities: BTreeMap<String, GroupIdentity> = BTreeMap::new();
    let mut group_memberships: BTreeMap<(GroupKey, String), BTreeSet<DataSplit>> =
        BTreeMap::new();
    let mut split_operations: BTreeMap<DataSplit, usize> = BTreeMap::new();
    let mut split_samples: BTreeMap<DataSplit, BTreeSet<String>> = BTreeMap::new();
    let mut split_views: BTreeMap<DataSplit, usize> = BTreeMap::new();
    let mut split_tasks: BTreeMap<DataSplit, BTreeMap<TaskRoute, usize>> =
        BTreeMap::new();
    let mut total_views = 0_usize;

    for operation in operations {
        validate_id("operation", &operation.operation_id)?;
        if !operation_ids.insert(operation.operation_id.clone()) {
            return Err(SplitAuditError::DuplicateIdentifier {
                kind: "operation",
                value: operation.operation_id.clone(),
            });
        }
        validate_sample_hash(&operation.sample_hash)?;
        validate_sha("label artifact", &operation.label_artifact_sha256)?;

        if !sample_task_pairs.insert((operation.sample_hash.clone(), operation.task_route)) {
            return Err(SplitAuditError::DuplicateOperationForTask {
                sample_hash: operation.sample_hash.clone(),
                task_route: operation.task_route,
            });
        }

        if let Some(previous_groups) = sample_group_identities.get(&operation.sample_hash) {
            if previous_groups != &operation.groups {
                return Err(SplitAuditError::SampleGroupIdentityMismatch {
                    sample_hash: operation.sample_hash.clone(),
                });
            }
        } else {
            sample_group_identities.insert(
                operation.sample_hash.clone(),
                operation.groups.clone(),
            );
        }

        if let Some(previous_split) = sample_splits.get(&operation.sample_hash) {
            if *previous_split != operation.split {
                return Err(SplitAuditError::SampleAppearsInMultipleSplits {
                    sample_hash: operation.sample_hash.clone(),
                    first: *previous_split,
                    second: operation.split,
                });
            }
        } else {
            sample_splits.insert(operation.sample_hash.clone(), operation.split);
        }

        *split_operations.entry(operation.split).or_default() += 1;
        split_samples
            .entry(operation.split)
            .or_default()
            .insert(operation.sample_hash.clone());
        *split_tasks
            .entry(operation.split)
            .or_default()
            .entry(operation.task_route)
            .or_default() += 1;

        for key in &policy.group_keys {
            let values = group_values(&operation.groups, *key);
            if values.is_empty() {
                return Err(SplitAuditError::MissingGroupKey {
                    operation_id: operation.operation_id.clone(),
                    group_key: *key,
                });
            }
            validate_unique_ids(
                "group identity",
                values.iter().map(String::as_str),
            )?;
            for value in values {
                validate_id("group identity", &value)?;
                group_memberships
                    .entry((*key, value))
                    .or_default()
                    .insert(operation.split);
            }
        }

        if operation.views.is_empty() {
            return Err(SplitAuditError::EmptyCollection("operation views"));
        }
        for view in &operation.views {
            validate_id("view", &view.view_id)?;
            if !view_ids.insert(view.view_id.clone()) {
                return Err(SplitAuditError::DuplicateIdentifier {
                    kind: "view",
                    value: view.view_id.clone(),
                });
            }
            validate_sample_hash(&view.sample_hash)?;
            validate_sha("view artifact", &view.artifact_sha256)?;

            if view.sample_hash != operation.sample_hash {
                return Err(SplitAuditError::ViewSampleMismatch {
                    operation_id: operation.operation_id.clone(),
                    view_id: view.view_id.clone(),
                });
            }
            if view.split != operation.split {
                return Err(SplitAuditError::ViewSplitMismatch {
                    operation_id: operation.operation_id.clone(),
                    view_id: view.view_id.clone(),
                });
            }
            if view.task_route != operation.task_route {
                return Err(SplitAuditError::ViewTaskMismatch {
                    operation_id: operation.operation_id.clone(),
                    view_id: view.view_id.clone(),
                });
            }
            if operation.task_route == TaskRoute::FalseMergeIdentification
                && view.view_kind == ViewKind::EmBest
            {
                return Err(SplitAuditError::TargetLeakingView {
                    operation_id: operation.operation_id.clone(),
                    view_id: view.view_id.clone(),
                });
            }
            if !policy.view_permissions.contains(&ViewPermission {
                task_route: operation.task_route,
                view_kind: view.view_kind,
            }) {
                return Err(SplitAuditError::MissingViewPermission {
                    task_route: operation.task_route,
                    view_kind: view.view_kind,
                });
            }

            *split_views.entry(operation.split).or_default() += 1;
            total_views += 1;
        }
    }

    let mut split_summaries = Vec::new();
    for split in [
        DataSplit::Train,
        DataSplit::Validation,
        DataSplit::Test,
        DataSplit::Adaptation,
        DataSplit::Calibration,
    ] {
        if let Some(operation_count) = split_operations.get(&split) {
            split_summaries.push(SplitSummary {
                split,
                operation_count: *operation_count,
                unique_sample_count: split_samples
                    .get(&split)
                    .map(BTreeSet::len)
                    .unwrap_or(0),
                view_instance_count: *split_views.get(&split).unwrap_or(&0),
                task_counts: split_tasks.get(&split).cloned().unwrap_or_default(),
            });
        }
    }

    let mut group_overlaps = Vec::new();
    for ((group_key, group_value), splits) in group_memberships {
        if splits.len() > 1 {
            group_overlaps.push(GroupOverlap {
                group_key,
                group_value,
                splits: splits.into_iter().collect(),
            });
        }
    }

    let status = if group_overlaps.is_empty() {
        SplitAuditStatus::GroupDisjoint
    } else {
        SplitAuditStatus::LeakageDetected
    };

    Ok(SplitAuditReport {
        report_version: "0.1",
        policy_id: policy.policy_id.clone(),
        operation_count: operations.len(),
        unique_sample_count: sample_splits.len(),
        view_instance_count: total_views,
        group_keys_audited: policy.group_keys.clone(),
        split_summaries,
        group_overlaps,
        status,
    })
}

fn validate_policy(policy: &SplitAuditPolicy) -> Result<(), SplitAuditError> {
    validate_id("policy", &policy.policy_id)?;
    if policy.group_keys.is_empty() {
        return Err(SplitAuditError::EmptyCollection("group_keys"));
    }
    if policy.view_permissions.is_empty() {
        return Err(SplitAuditError::EmptyCollection("view_permissions"));
    }
    let mut group_keys = BTreeSet::new();
    for key in &policy.group_keys {
        if !group_keys.insert(*key) {
            return Err(SplitAuditError::DuplicateIdentifier {
                kind: "group key",
                value: format!("{key:?}"),
            });
        }
    }
    let mut view_permissions = BTreeSet::new();
    for permission in &policy.view_permissions {
        if !view_permissions.insert(*permission) {
            return Err(SplitAuditError::DuplicateIdentifier {
                kind: "view permission",
                value: format!("{:?}:{:?}", permission.task_route, permission.view_kind),
            });
        }
    }
    Ok(())
}

fn group_values(groups: &GroupIdentity, key: GroupKey) -> Vec<String> {
    match key {
        GroupKey::SourceArchive => groups.source_archive_id.iter().cloned().collect(),
        GroupKey::SourceVolume => groups.source_volume_id.iter().cloned().collect(),
        GroupKey::SourceOperation => groups.source_operation_id.iter().cloned().collect(),
        GroupKey::SpatialCube => groups.spatial_cube_id.iter().cloned().collect(),
        GroupKey::Neuron => groups.neuron_ids.clone(),
        GroupKey::Animal => groups.animal_id.iter().cloned().collect(),
        GroupKey::Session => groups.session_id.iter().cloned().collect(),
        GroupKey::Species => groups.species.iter().cloned().collect(),
        GroupKey::Workflow => groups.workflow_id.iter().cloned().collect(),
    }
}

fn validate_sample_hash(value: &str) -> Result<(), SplitAuditError> {
    if value.len() != 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SplitAuditError::InvalidSampleHash(value.to_owned()));
    }
    Ok(())
}

fn validate_sha(kind: &'static str, value: &str) -> Result<(), SplitAuditError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SplitAuditError::InvalidSha256 {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_unique_ids<'a>(
    kind: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), SplitAuditError> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_id(kind, value)?;
        if !seen.insert(value.to_owned()) {
            return Err(SplitAuditError::DuplicateIdentifier {
                kind,
                value: value.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), SplitAuditError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(SplitAuditError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn groups(volume: &str, neuron: &str, species: &str) -> GroupIdentity {
        GroupIdentity {
            source_archive_id: Some("archive-001".to_owned()),
            source_volume_id: Some(volume.to_owned()),
            source_operation_id: Some(format!("source-op-{neuron}")),
            spatial_cube_id: Some(format!("cube-{volume}-{neuron}")),
            neuron_ids: vec![neuron.to_owned()],
            animal_id: Some("animal-001".to_owned()),
            session_id: Some("session-001".to_owned()),
            species: Some(species.to_owned()),
            workflow_id: Some("workflow-v1".to_owned()),
        }
    }

    fn view(
        id: &str,
        sample_hash: &str,
        split: DataSplit,
        task: TaskRoute,
        kind: ViewKind,
    ) -> ViewInstance {
        ViewInstance {
            view_id: id.to_owned(),
            sample_hash: sample_hash.to_owned(),
            split,
            task_route: task,
            view_kind: kind,
            artifact_sha256: SHA_B.to_owned(),
        }
    }

    fn operation(
        id: &str,
        sample_hash: &str,
        split: DataSplit,
        task: TaskRoute,
        volume: &str,
        neuron: &str,
        species: &str,
        view_kind: ViewKind,
    ) -> OperationRecord {
        OperationRecord {
            operation_id: id.to_owned(),
            sample_hash: sample_hash.to_owned(),
            split,
            task_route: task,
            groups: groups(volume, neuron, species),
            label_artifact_sha256: SHA_C.to_owned(),
            views: vec![view(
                &format!("view-{id}"),
                sample_hash,
                split,
                task,
                view_kind,
            )],
        }
    }

    fn policy(group_keys: Vec<GroupKey>) -> SplitAuditPolicy {
        SplitAuditPolicy {
            policy_id: "leave-volume-out-v1".to_owned(),
            group_keys,
            view_permissions: vec![
                ViewPermission {
                    task_route: TaskRoute::FalseSplitCorrection,
                    view_kind: ViewKind::GeometryDual,
                },
                ViewPermission {
                    task_route: TaskRoute::FalseMergeIdentification,
                    view_kind: ViewKind::GeometrySingle,
                },
                ViewPermission {
                    task_route: TaskRoute::FalseSplitCorrection,
                    view_kind: ViewKind::EmBest,
                },
                ViewPermission {
                    task_route: TaskRoute::FalseMergeIdentification,
                    view_kind: ViewKind::EmXy,
                },
            ],
        }
    }

    #[test]
    fn group_disjoint_manifest_is_clean_and_counts_operations_samples_and_views() {
        let operations = vec![
            operation(
                "op-train-1",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "volume-train",
                "neuron-train",
                "mouse",
                ViewKind::GeometryDual,
            ),
            operation(
                "op-val-1",
                "00000000000000000000000000000002",
                DataSplit::Validation,
                TaskRoute::FalseSplitCorrection,
                "volume-val",
                "neuron-val",
                "mouse",
                ViewKind::GeometryDual,
            ),
            operation(
                "op-test-1",
                "00000000000000000000000000000003",
                DataSplit::Test,
                TaskRoute::FalseMergeIdentification,
                "volume-test",
                "neuron-test",
                "fly",
                ViewKind::EmXy,
            ),
        ];

        let report = audit_splits(
            &policy(vec![GroupKey::SourceVolume]),
            &operations,
        )
        .unwrap();

        assert_eq!(report.status, SplitAuditStatus::GroupDisjoint);
        assert_eq!(report.operation_count, 3);
        assert_eq!(report.unique_sample_count, 3);
        assert_eq!(report.view_instance_count, 3);
        assert_eq!(report.split_summaries.len(), 3);
    }

    #[test]
    fn overlapping_source_volume_is_reported_as_leakage_not_clean() {
        let operations = vec![
            operation(
                "op-train",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "shared-volume",
                "neuron-a",
                "mouse",
                ViewKind::GeometryDual,
            ),
            operation(
                "op-test",
                "00000000000000000000000000000002",
                DataSplit::Test,
                TaskRoute::FalseSplitCorrection,
                "shared-volume",
                "neuron-b",
                "mouse",
                ViewKind::GeometryDual,
            ),
        ];

        let report = audit_splits(
            &policy(vec![GroupKey::SourceVolume]),
            &operations,
        )
        .unwrap();

        assert_eq!(report.status, SplitAuditStatus::LeakageDetected);
        assert_eq!(report.group_overlaps.len(), 1);
        assert_eq!(report.group_overlaps[0].group_value, "shared-volume");
        assert_eq!(
            report.group_overlaps[0].splits,
            vec![DataSplit::Train, DataSplit::Test]
        );
    }

    #[test]
    fn same_sample_cannot_cross_splits_even_for_different_task_routes() {
        let operations = vec![
            operation(
                "op-a",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "volume-a",
                "neuron-a",
                "mouse",
                ViewKind::GeometryDual,
            ),
            operation(
                "op-b",
                "00000000000000000000000000000001",
                DataSplit::Test,
                TaskRoute::FalseMergeIdentification,
                "volume-a",
                "neuron-a",
                "mouse",
                ViewKind::EmXy,
            ),
        ];

        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &operations),
            Err(SplitAuditError::SampleAppearsInMultipleSplits { .. })
        ));
    }

    #[test]
    fn same_sample_can_have_two_task_routes_inside_one_split() {
        let operations = vec![
            operation(
                "op-split",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "volume-a",
                "neuron-a",
                "mouse",
                ViewKind::GeometryDual,
            ),
            operation(
                "op-merge",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseMergeIdentification,
                "volume-a",
                "neuron-a",
                "mouse",
                ViewKind::EmXy,
            ),
        ];

        let report = audit_splits(
            &policy(vec![GroupKey::SourceVolume]),
            &operations,
        )
        .unwrap();

        assert_eq!(report.operation_count, 2);
        assert_eq!(report.unique_sample_count, 1);
        assert_eq!(report.view_instance_count, 2);
        assert_eq!(report.status, SplitAuditStatus::GroupDisjoint);
    }

    #[test]
    fn same_sample_task_routes_must_have_identical_source_group_identity() {
        let first = operation(
            "op-split",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        let mut second = operation(
            "op-merge",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseMergeIdentification,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::EmXy,
        );
        second.groups.source_operation_id = Some("different-source-operation".to_owned());

        assert_eq!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[first, second]),
            Err(SplitAuditError::SampleGroupIdentityMismatch {
                sample_hash: "00000000000000000000000000000001".to_owned(),
            })
        );
    }

    #[test]
    fn duplicate_operation_id_is_rejected() {
        let first = operation(
            "same-op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        let mut second = operation(
            "same-op",
            "00000000000000000000000000000002",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-b",
            "mouse",
            ViewKind::GeometryDual,
        );
        second.operation_id = first.operation_id.clone();

        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[first, second]),
            Err(SplitAuditError::DuplicateIdentifier { kind: "operation", .. })
        ));
    }

    #[test]
    fn duplicate_sample_task_pair_is_rejected() {
        let first = operation(
            "op-1",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        let second = operation(
            "op-2",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[first, second]),
            Err(SplitAuditError::DuplicateOperationForTask { .. })
        ));
    }

    #[test]
    fn same_neuron_across_splits_is_detected_even_when_volumes_differ() {
        let operations = vec![
            operation(
                "op-train",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "volume-a",
                "shared-neuron",
                "mouse",
                ViewKind::GeometryDual,
            ),
            operation(
                "op-test",
                "00000000000000000000000000000002",
                DataSplit::Test,
                TaskRoute::FalseSplitCorrection,
                "volume-b",
                "shared-neuron",
                "mouse",
                ViewKind::GeometryDual,
            ),
        ];
        let report = audit_splits(
            &policy(vec![GroupKey::Neuron]),
            &operations,
        )
        .unwrap();
        assert_eq!(report.status, SplitAuditStatus::LeakageDetected);
        assert_eq!(report.group_overlaps[0].group_key, GroupKey::Neuron);
    }

    #[test]
    fn multiple_neuron_ids_are_all_checked_for_overlap() {
        let mut train = operation(
            "op-train",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        train.groups.neuron_ids = vec!["neuron-a".to_owned(), "neuron-shared".to_owned()];
        let test = operation(
            "op-test",
            "00000000000000000000000000000002",
            DataSplit::Test,
            TaskRoute::FalseSplitCorrection,
            "volume-b",
            "neuron-shared",
            "mouse",
            ViewKind::GeometryDual,
        );

        let report = audit_splits(&policy(vec![GroupKey::Neuron]), &[train, test]).unwrap();
        assert_eq!(report.status, SplitAuditStatus::LeakageDetected);
        assert_eq!(report.group_overlaps[0].group_value, "neuron-shared");
    }

    #[test]
    fn source_operation_lineage_cannot_cross_group_held_out_splits() {
        let mut train = operation(
            "op-train",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        let mut test = operation(
            "op-test",
            "00000000000000000000000000000002",
            DataSplit::Test,
            TaskRoute::FalseSplitCorrection,
            "volume-b",
            "neuron-b",
            "mouse",
            ViewKind::GeometryDual,
        );
        train.groups.source_operation_id = Some("upstream-edit-42".to_owned());
        test.groups.source_operation_id = Some("upstream-edit-42".to_owned());

        let report = audit_splits(
            &policy(vec![GroupKey::SourceOperation]),
            &[train, test],
        )
        .unwrap();

        assert_eq!(report.status, SplitAuditStatus::LeakageDetected);
        assert_eq!(report.group_overlaps[0].group_key, GroupKey::SourceOperation);
        assert_eq!(report.group_overlaps[0].group_value, "upstream-edit-42");
    }

    #[test]
    fn spatial_cube_identity_is_checked_across_splits() {
        let mut train = operation(
            "op-train",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        let mut test = operation(
            "op-test",
            "00000000000000000000000000000002",
            DataSplit::Test,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-b",
            "mouse",
            ViewKind::GeometryDual,
        );
        train.groups.spatial_cube_id = Some("cube-50um-001".to_owned());
        test.groups.spatial_cube_id = Some("cube-50um-001".to_owned());

        let report = audit_splits(
            &policy(vec![GroupKey::SpatialCube]),
            &[train, test],
        )
        .unwrap();

        assert_eq!(report.status, SplitAuditStatus::LeakageDetected);
        assert_eq!(report.group_overlaps[0].group_key, GroupKey::SpatialCube);
        assert_eq!(report.group_overlaps[0].group_value, "cube-50um-001");
    }

    #[test]
    fn missing_requested_group_key_is_not_auditable() {
        let mut op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        op.groups.animal_id = None;
        assert_eq!(
            audit_splits(&policy(vec![GroupKey::Animal]), &[op]),
            Err(SplitAuditError::MissingGroupKey {
                operation_id: "op".to_owned(),
                group_key: GroupKey::Animal,
            })
        );
    }

    #[test]
    fn hard_rejects_em_best_for_false_merge_even_if_permission_exists() {
        let mut p = policy(vec![GroupKey::SourceVolume]);
        p.view_permissions.push(ViewPermission {
            task_route: TaskRoute::FalseMergeIdentification,
            view_kind: ViewKind::EmBest,
        });
        let op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseMergeIdentification,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::EmBest,
        );
        assert!(matches!(
            audit_splits(&p, &[op]),
            Err(SplitAuditError::TargetLeakingView { .. })
        ));
    }

    #[test]
    fn view_must_inherit_sample_split_and_task_from_parent_operation() {
        let mut op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        op.views[0].split = DataSplit::Test;
        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[op]),
            Err(SplitAuditError::ViewSplitMismatch { .. })
        ));
    }

    #[test]
    fn view_must_inherit_sample_hash_from_parent_operation() {
        let mut op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        op.views[0].sample_hash = "00000000000000000000000000000002".to_owned();

        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[op]),
            Err(SplitAuditError::ViewSampleMismatch { .. })
        ));
    }

    #[test]
    fn view_must_inherit_task_route_from_parent_operation() {
        let mut op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        op.views[0].task_route = TaskRoute::FalseMergeIdentification;

        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[op]),
            Err(SplitAuditError::ViewTaskMismatch { .. })
        ));
    }

    #[test]
    fn report_is_invariant_to_operation_input_order() {
        let first = operation(
            "op-train",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-train",
            "neuron-train",
            "mouse",
            ViewKind::GeometryDual,
        );
        let second = operation(
            "op-val",
            "00000000000000000000000000000002",
            DataSplit::Validation,
            TaskRoute::FalseSplitCorrection,
            "volume-val",
            "neuron-val",
            "mouse",
            ViewKind::GeometryDual,
        );
        let third = operation(
            "op-test",
            "00000000000000000000000000000003",
            DataSplit::Test,
            TaskRoute::FalseMergeIdentification,
            "volume-test",
            "neuron-test",
            "fly",
            ViewKind::EmXy,
        );

        let forward = audit_splits(
            &policy(vec![GroupKey::SourceVolume, GroupKey::Neuron]),
            &[first.clone(), second.clone(), third.clone()],
        )
        .unwrap();
        let reverse = audit_splits(
            &policy(vec![GroupKey::SourceVolume, GroupKey::Neuron]),
            &[third, second, first],
        )
        .unwrap();

        assert_eq!(forward, reverse);
    }

    #[test]
    fn view_must_be_permitted_for_task() {
        let op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::EmXy,
        );
        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[op]),
            Err(SplitAuditError::MissingViewPermission { .. })
        ));
    }

    #[test]
    fn duplicate_view_id_is_rejected_globally() {
        let a = operation(
            "op-a",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        let mut b = operation(
            "op-b",
            "00000000000000000000000000000002",
            DataSplit::Test,
            TaskRoute::FalseSplitCorrection,
            "volume-b",
            "neuron-b",
            "mouse",
            ViewKind::GeometryDual,
        );
        b.views[0].view_id = a.views[0].view_id.clone();

        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[a, b]),
            Err(SplitAuditError::DuplicateIdentifier { kind: "view", .. })
        ));
    }

    #[test]
    fn empty_operations_group_keys_and_permissions_fail_closed() {
        assert_eq!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[]),
            Err(SplitAuditError::EmptyCollection("operations"))
        );
        let mut p = policy(vec![]);
        assert_eq!(
            audit_splits(&p, &[operation(
                "op",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "volume-a",
                "neuron-a",
                "mouse",
                ViewKind::GeometryDual,
            )]),
            Err(SplitAuditError::EmptyCollection("group_keys"))
        );
        p = policy(vec![GroupKey::SourceVolume]);
        p.view_permissions.clear();
        assert_eq!(
            audit_splits(&p, &[operation(
                "op",
                "00000000000000000000000000000001",
                DataSplit::Train,
                TaskRoute::FalseSplitCorrection,
                "volume-a",
                "neuron-a",
                "mouse",
                ViewKind::GeometryDual,
            )]),
            Err(SplitAuditError::EmptyCollection("view_permissions"))
        );
    }

    #[test]
    fn duplicate_group_key_and_view_permission_fail_closed() {
        let mut p = policy(vec![GroupKey::SourceVolume, GroupKey::SourceVolume]);
        let op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        assert!(matches!(
            audit_splits(&p, &[op.clone()]),
            Err(SplitAuditError::DuplicateIdentifier { kind: "group key", .. })
        ));

        p = policy(vec![GroupKey::SourceVolume]);
        p.view_permissions.push(p.view_permissions[0]);
        assert!(matches!(
            audit_splits(&p, &[op]),
            Err(SplitAuditError::DuplicateIdentifier { kind: "view permission", .. })
        ));
    }

    #[test]
    fn sample_and_sha_formats_are_validated() {
        let mut op = operation(
            "op",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        op.sample_hash = "not-a-sample-hash".to_owned();
        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[op.clone()]),
            Err(SplitAuditError::InvalidSampleHash(_))
        ));

        op.sample_hash = "00000000000000000000000000000001".to_owned();
        op.views[0].artifact_sha256 = "short".to_owned();
        assert!(matches!(
            audit_splits(&policy(vec![GroupKey::SourceVolume]), &[op]),
            Err(SplitAuditError::InvalidSha256 { .. })
        ));
    }

    #[test]
    fn report_counts_two_task_routes_and_all_derived_views() {
        let mut first = operation(
            "op-split",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseSplitCorrection,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::GeometryDual,
        );
        first.views.push(view(
            "view-op-split-emxy",
            &first.sample_hash,
            first.split,
            first.task_route,
            ViewKind::EmBest,
        ));
        let second = operation(
            "op-merge",
            "00000000000000000000000000000001",
            DataSplit::Train,
            TaskRoute::FalseMergeIdentification,
            "volume-a",
            "neuron-a",
            "mouse",
            ViewKind::EmXy,
        );

        let report = audit_splits(
            &policy(vec![GroupKey::SourceVolume]),
            &[first, second],
        )
        .unwrap();

        assert_eq!(report.operation_count, 2);
        assert_eq!(report.unique_sample_count, 1);
        assert_eq!(report.view_instance_count, 3);
        assert_eq!(report.split_summaries[0].operation_count, 2);
        assert_eq!(report.split_summaries[0].view_instance_count, 3);
    }
}
