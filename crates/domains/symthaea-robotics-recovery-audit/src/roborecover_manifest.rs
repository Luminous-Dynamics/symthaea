//! Strict importer for the pinned RoboRecover split CSV manifests.
//!
//! The CSV provides scenario IDs, relative artifact paths, scenario/original
//! trajectory SHA-256 values, action-prefix lengths, task names and Stage/Type
//! metadata. It does not provide construction-source labels or scenario bytes.
//! Those must not be inferred from this file.

use std::collections::{BTreeMap, BTreeSet};

use super::roborecover_score::{
    BenchmarkPlatform, BenchmarkSplit, DeviationType, ExecutionMode, ManifestIssue,
    PlannedScenario, RecoveryStage, RoboRecoverManifest, ScenarioSource, StageTypeGroup,
};

pub const ROBORECOVER_SOURCE_REPOSITORY: &str = "RUCKBReasoning/RoboRecover";
pub const ROBORECOVER_SOURCE_REVISION: &str =
    "089b9420e8b7c303e2338a8212a8b6c27c26a79b";
const EXPECTED_HEADER: &str =
    "sample_id,path,sha256,original_sha256,actions,replay_prefix_steps,task_name,s_stage,d_stage";

const TEST_MACRO9: [StageTypeGroup; 9] = [
    StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::WrongPose },
    StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::WrongTarget },
    StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::StageRollback },
    StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::SceneChange },
    StageTypeGroup { stage: RecoveryStage::Contact, deviation: DeviationType::WrongPose },
    StageTypeGroup { stage: RecoveryStage::Move, deviation: DeviationType::WrongPose },
    StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::WrongPose },
    StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::WrongTarget },
    StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::SceneChange },
];

/// Git blob SHA returned for each CSV at the pinned upstream revision.
pub const fn pinned_split_blob_sha(
    platform: BenchmarkPlatform,
    split: BenchmarkSplit,
) -> &'static str {
    match (platform, split) {
        (BenchmarkPlatform::Libero, BenchmarkSplit::Train) =>
            "4b4caf53d03ccdad5883cf556aab8353bcdc6542",
        (BenchmarkPlatform::Libero, BenchmarkSplit::Test) =>
            "925af3d5ab5e810cd65b86f6d4f35bbc3891d67f",
        (BenchmarkPlatform::RoboTwin, BenchmarkSplit::Train) =>
            "b211728e96d4e6548d4d5353daf73346c37f5a4d",
        (BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test) =>
            "c9b40e12d6f8d2dab723815dfc2e966ea7e62cd5",
    }
}

const fn expected_rows(split: BenchmarkSplit) -> usize {
    match split {
        BenchmarkSplit::Train => 800,
        BenchmarkSplit::Test => 200,
    }
}

const fn platform_dir(platform: BenchmarkPlatform) -> &'static str {
    match platform {
        BenchmarkPlatform::Libero => "libero",
        BenchmarkPlatform::RoboTwin => "robotwin",
    }
}

const fn split_dir(split: BenchmarkSplit) -> &'static str {
    match split {
        BenchmarkSplit::Train => "train",
        BenchmarkSplit::Test => "test",
    }
}

/// All identities needed to turn one pinned upstream split CSV into a score manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitImportContext {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    pub source_repository: String,
    pub source_revision: String,
    pub split_manifest_blob_sha: String,
    pub policy_digest: String,
    pub evaluator_digest: String,
    pub arm_id: String,
}

/// A reproducible import result. Scenario source is Unknown because the upstream
/// split CSV does not encode Natural / ActionPerturbed / HumanConstructed labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedSplitManifest {
    pub manifest: RoboRecoverManifest,
    pub row_count: usize,
    pub group_counts: BTreeMap<StageTypeGroup, usize>,
    pub source_csv_header: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SplitManifestIssue {
    EmptyInput,
    HeaderMismatch,
    QuotedFieldUnsupported { line: usize },
    BlankLine { line: usize },
    WrongColumnCount { line: usize, actual: usize },
    BlankField { line: usize, field: &'static str },
    InvalidSampleId { line: usize },
    InvalidPath { line: usize, path: String },
    SampleIdPathMismatch { line: usize },
    InvalidSha256 { line: usize, field: &'static str },
    InvalidActionCount { line: usize, field: &'static str },
    ActionPrefixMismatch { line: usize },
    UnknownStage { line: usize, value: String },
    UnknownDeviation { line: usize, value: String },
    DuplicateSampleId { line: usize, sample_id: String },
    DuplicatePath { line: usize, path: String },
    WrongScenarioCount { expected: usize, actual: usize },
    MissingTestMacro9Group(StageTypeGroup),
    UnexpectedTestMacro9Group(StageTypeGroup),
    TestGroupCountMismatch {
        group: StageTypeGroup,
        expected: usize,
        actual: usize,
    },
    WrongRepository,
    WrongSourceRevision,
    WrongSplitManifestBlobSha,
    InvalidScoreManifest(Vec<ManifestIssue>),
}

fn is_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_stage(value: &str) -> Option<RecoveryStage> {
    match value {
        "Approach" => Some(RecoveryStage::Approach),
        "Contact" => Some(RecoveryStage::Contact),
        "Move" => Some(RecoveryStage::Move),
        "Place" => Some(RecoveryStage::Place),
        _ => None,
    }
}

fn parse_deviation(value: &str) -> Option<DeviationType> {
    match value {
        "Wrong Target" => Some(DeviationType::WrongTarget),
        "Wrong Pose" => Some(DeviationType::WrongPose),
        "Stage Rollback" => Some(DeviationType::StageRollback),
        "Scene Change" => Some(DeviationType::SceneChange),
        _ => None,
    }
}

fn valid_sample_id(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
        })
}

fn expected_test_group_count(
    platform: BenchmarkPlatform,
    group: StageTypeGroup,
) -> Option<usize> {
    use DeviationType::{SceneChange, StageRollback, WrongPose, WrongTarget};
    use RecoveryStage::{Approach, Contact, Move, Place};

    match (platform, group.stage, group.deviation) {
        (BenchmarkPlatform::Libero, Approach, WrongPose) => Some(49),
        (BenchmarkPlatform::Libero, Approach, WrongTarget) => Some(25),
        (BenchmarkPlatform::Libero, Approach, StageRollback) => Some(21),
        (BenchmarkPlatform::Libero, Approach, SceneChange) => Some(28),
        (BenchmarkPlatform::Libero, Contact, WrongPose) => Some(12),
        (BenchmarkPlatform::Libero, Move, WrongPose) => Some(16),
        (BenchmarkPlatform::Libero, Place, WrongPose) => Some(31),
        (BenchmarkPlatform::Libero, Place, WrongTarget) => Some(7),
        (BenchmarkPlatform::Libero, Place, SceneChange) => Some(11),
        (BenchmarkPlatform::RoboTwin, Approach, WrongPose) => Some(44),
        (BenchmarkPlatform::RoboTwin, Approach, WrongTarget) => Some(29),
        (BenchmarkPlatform::RoboTwin, Approach, StageRollback) => Some(24),
        (BenchmarkPlatform::RoboTwin, Approach, SceneChange) => Some(4),
        (BenchmarkPlatform::RoboTwin, Contact, WrongPose) => Some(21),
        (BenchmarkPlatform::RoboTwin, Move, WrongPose) => Some(35),
        (BenchmarkPlatform::RoboTwin, Place, WrongPose) => Some(23),
        (BenchmarkPlatform::RoboTwin, Place, WrongTarget) => Some(18),
        (BenchmarkPlatform::RoboTwin, Place, SceneChange) => Some(2),
        _ => None,
    }
}

fn expected_path(
    platform: BenchmarkPlatform,
    split: BenchmarkSplit,
    sample_id: &str,
) -> String {
    format!(
        "{}/{}/{}.json",
        platform_dir(platform),
        split_dir(split),
        sample_id
    )
}

/// Parse and validate a RoboRecover CSV at the pinned source revision.
///
/// This parser deliberately accepts only the exact nine-column source schema,
/// unquoted single-line rows, pinned Git blob identity and expected split size.
/// The source SHA-256 fields identify scenario artifacts; this function does not
/// claim to have downloaded or re-hashed those JSON artifacts.
pub fn parse_pinned_split_csv(
    csv_text: &str,
    context: SplitImportContext,
) -> Result<ParsedSplitManifest, Vec<SplitManifestIssue>> {
    let mut issues = Vec::new();

    if context.source_repository != ROBORECOVER_SOURCE_REPOSITORY {
        issues.push(SplitManifestIssue::WrongRepository);
    }
    if context.source_revision != ROBORECOVER_SOURCE_REVISION {
        issues.push(SplitManifestIssue::WrongSourceRevision);
    }
    if context.split_manifest_blob_sha != pinned_split_blob_sha(context.platform, context.split) {
        issues.push(SplitManifestIssue::WrongSplitManifestBlobSha);
    }

    let mut lines = csv_text.split_terminator('\n');
    let Some(header_line) = lines.next() else {
        issues.push(SplitManifestIssue::EmptyInput);
        return Err(issues);
    };
    let header = header_line.trim_end_matches('\r');
    if header != EXPECTED_HEADER {
        issues.push(SplitManifestIssue::HeaderMismatch);
    }

    let mut scenarios = Vec::new();
    let mut group_counts = BTreeMap::<StageTypeGroup, usize>::new();
    let mut sample_ids = BTreeSet::new();
    let mut paths = BTreeSet::new();

    for (zero_index, raw_line) in lines.enumerate() {
        let line_number = zero_index + 2;
        let line = raw_line.trim_end_matches('\r');

        if line.is_empty() {
            issues.push(SplitManifestIssue::BlankLine { line: line_number });
            continue;
        }
        if line.contains('"') {
            issues.push(SplitManifestIssue::QuotedFieldUnsupported { line: line_number });
            continue;
        }

        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() != 9 {
            issues.push(SplitManifestIssue::WrongColumnCount {
                line: line_number,
                actual: fields.len(),
            });
            continue;
        }

        const FIELD_NAMES: [&str; 9] = [
            "sample_id",
            "path",
            "sha256",
            "original_sha256",
            "actions",
            "replay_prefix_steps",
            "task_name",
            "s_stage",
            "d_stage",
        ];
        for (index, value) in fields.iter().enumerate() {
            if value.trim().is_empty() {
                issues.push(SplitManifestIssue::BlankField {
                    line: line_number,
                    field: FIELD_NAMES[index],
                });
            }
        }
        if fields.iter().any(|value| value.trim().is_empty()) {
            continue;
        }

        let sample_id = fields[0];
        let path = fields[1];
        let scenario_sha256 = fields[2];
        let original_sha256 = fields[3];
        let actions = fields[4].parse::<u32>().ok();
        let replay_prefix_steps = fields[5].parse::<u32>().ok();
        let task_name = fields[6];

        if !valid_sample_id(sample_id) {
            issues.push(SplitManifestIssue::InvalidSampleId { line: line_number });
            continue;
        }

        if path.starts_with('/')
            || path.contains('\\')
            || path.split('/').any(|part| part == ".." || part == ".")
            || path != expected_path(context.platform, context.split, sample_id)
        {
            issues.push(SplitManifestIssue::InvalidPath {
                line: line_number,
                path: path.to_owned(),
            });
            continue;
        }

        if !path.ends_with(&format!("/{sample_id}.json")) {
            issues.push(SplitManifestIssue::SampleIdPathMismatch { line: line_number });
            continue;
        }

        if !is_hex_digest(scenario_sha256, 64) {
            issues.push(SplitManifestIssue::InvalidSha256 {
                line: line_number,
                field: "sha256",
            });
            continue;
        }
        if !is_hex_digest(original_sha256, 64) {
            issues.push(SplitManifestIssue::InvalidSha256 {
                line: line_number,
                field: "original_sha256",
            });
            continue;
        }

        let Some(actions) = actions.filter(|value| *value > 0) else {
            issues.push(SplitManifestIssue::InvalidActionCount {
                line: line_number,
                field: "actions",
            });
            continue;
        };
        let Some(replay_prefix_steps) = replay_prefix_steps.filter(|value| *value > 0) else {
            issues.push(SplitManifestIssue::InvalidActionCount {
                line: line_number,
                field: "replay_prefix_steps",
            });
            continue;
        };
        if actions != replay_prefix_steps {
            issues.push(SplitManifestIssue::ActionPrefixMismatch { line: line_number });
            continue;
        }

        let Some(stage) = parse_stage(fields[7]) else {
            issues.push(SplitManifestIssue::UnknownStage {
                line: line_number,
                value: fields[7].to_owned(),
            });
            continue;
        };
        let Some(deviation) = parse_deviation(fields[8]) else {
            issues.push(SplitManifestIssue::UnknownDeviation {
                line: line_number,
                value: fields[8].to_owned(),
            });
            continue;
        };

        let duplicate_sample_id = !sample_ids.insert(sample_id.to_owned());
        let duplicate_path = !paths.insert(path.to_owned());
        if duplicate_sample_id {
            issues.push(SplitManifestIssue::DuplicateSampleId {
                line: line_number,
                sample_id: sample_id.to_owned(),
            });
        }
        if duplicate_path {
            issues.push(SplitManifestIssue::DuplicatePath {
                line: line_number,
                path: path.to_owned(),
            });
        }
        if duplicate_sample_id || duplicate_path {
            continue;
        }

        let group = StageTypeGroup { stage, deviation };
        *group_counts.entry(group).or_insert(0) += 1;

        scenarios.push(PlannedScenario {
            scenario_id: sample_id.to_owned(),
            source_path: path.to_owned(),
            scenario_content_sha256: scenario_sha256.to_owned(),
            original_trajectory_sha256: original_sha256.to_owned(),
            task_name: task_name.to_owned(),
            actions,
            replay_prefix_steps,
            source: ScenarioSource::Unknown,
            stage,
            deviation,
            expected_trial_ids: vec![sample_id.to_owned()],
        });
    }

    let expected = expected_rows(context.split);
    if scenarios.len() != expected {
        issues.push(SplitManifestIssue::WrongScenarioCount {
            expected,
            actual: scenarios.len(),
        });
    }

    let macro9_groups = if context.split == BenchmarkSplit::Test {
        let expected_groups = TEST_MACRO9.iter().copied().collect::<BTreeSet<_>>();
        let observed_groups = group_counts.keys().copied().collect::<BTreeSet<_>>();

        for group in expected_groups.difference(&observed_groups) {
            issues.push(SplitManifestIssue::MissingTestMacro9Group(*group));
        }
        for group in observed_groups.difference(&expected_groups) {
            issues.push(SplitManifestIssue::UnexpectedTestMacro9Group(*group));
        }
        for group in &TEST_MACRO9 {
            let expected_count = expected_test_group_count(context.platform, *group)
                .expect("every frozen Macro-9 group has a platform count");
            let actual_count = group_counts.get(group).copied().unwrap_or(0);
            if actual_count != expected_count {
                issues.push(SplitManifestIssue::TestGroupCountMismatch {
                    group: *group,
                    expected: expected_count,
                    actual: actual_count,
                });
            }
        }
        TEST_MACRO9.to_vec()
    } else {
        Vec::new()
    };

    if !issues.is_empty() {
        return Err(issues);
    }

    let manifest = RoboRecoverManifest {
        platform: context.platform,
        split: context.split,
        execution_mode: ExecutionMode::ReplayThenInfer,
        source_repository: context.source_repository,
        source_revision: context.source_revision,
        split_manifest_blob_sha: context.split_manifest_blob_sha,
        policy_digest: context.policy_digest,
        evaluator_digest: context.evaluator_digest,
        arm_id: context.arm_id,
        scenarios,
        macro9_groups,
    };

    if let Err(validation_issues) = super::roborecover_score::validate_manifest(&manifest) {
        return Err(vec![SplitManifestIssue::InvalidScoreManifest(validation_issues)]);
    }

    Ok(ParsedSplitManifest {
        manifest,
        row_count: expected,
        group_counts,
        source_csv_header: header.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(platform: BenchmarkPlatform, split: BenchmarkSplit) -> SplitImportContext {
        SplitImportContext {
            platform,
            split,
            source_repository: ROBORECOVER_SOURCE_REPOSITORY.to_owned(),
            source_revision: ROBORECOVER_SOURCE_REVISION.to_owned(),
            split_manifest_blob_sha: pinned_split_blob_sha(platform, split).to_owned(),
            policy_digest: "a".repeat(64),
            evaluator_digest: "b".repeat(64),
            arm_id: "fixture-policy".to_owned(),
        }
    }

    fn row(
        id: &str,
        platform: BenchmarkPlatform,
        split: BenchmarkSplit,
        stage: &str,
        deviation: &str,
    ) -> String {
        let file_hash = "c".repeat(64);
        let original_hash = "d".repeat(64);
        format!(
            "{id},{}/{}/{}.json,{file_hash},{original_hash},12,12,fixture-task,{stage},{deviation}",
            platform_dir(platform),
            split_dir(split),
            id
        )
    }

    fn synthetic_csv(
        platform: BenchmarkPlatform,
        split: BenchmarkSplit,
        include_all_test_groups: bool,
    ) -> String {
        let mut rows = vec![EXPECTED_HEADER.to_owned()];
        let groups = if include_all_test_groups {
            TEST_MACRO9.to_vec()
        } else {
            vec![StageTypeGroup {
                stage: RecoveryStage::Approach,
                deviation: DeviationType::WrongPose,
            }]
        };

        let assigned_groups = if split == BenchmarkSplit::Test && include_all_test_groups {
            let mut assigned = Vec::new();
            for group in &TEST_MACRO9 {
                let count = expected_test_group_count(platform, *group).unwrap();
                assigned.extend(std::iter::repeat(*group).take(count));
            }
            assigned
        } else {
            (0..expected_rows(split))
                .map(|index| groups[index % groups.len()])
                .collect()
        };

        for (index, group) in assigned_groups.into_iter().enumerate() {
            let stage = match group.stage {
                RecoveryStage::Approach => "Approach",
                RecoveryStage::Contact => "Contact",
                RecoveryStage::Move => "Move",
                RecoveryStage::Place => "Place",
            };
            let deviation = match group.deviation {
                DeviationType::WrongTarget => "Wrong Target",
                DeviationType::WrongPose => "Wrong Pose",
                DeviationType::StageRollback => "Stage Rollback",
                DeviationType::SceneChange => "Scene Change",
            };
            rows.push(row(
                &format!("sample-{index:03}"),
                platform,
                split,
                stage,
                deviation,
            ));
        }
        rows.join("\n") + "\n"
    }

    #[test]
    fn pinned_test_split_imports_two_hundred_unique_scenarios_and_nine_groups() {
        let csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        let parsed = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap();

        assert_eq!(parsed.row_count, 200);
        assert_eq!(parsed.manifest.scenarios.len(), 200);
        assert_eq!(parsed.manifest.macro9_groups.len(), 9);
        assert_eq!(parsed.group_counts.len(), 9);
        assert_eq!(parsed.manifest.scenarios[0].source, ScenarioSource::Unknown);
        assert!(super::super::roborecover_score::validate_manifest(&parsed.manifest).is_ok());
    }

    #[test]
    fn pinned_training_split_imports_eight_hundred_rows_without_macro9() {
        let csv = synthetic_csv(BenchmarkPlatform::Libero, BenchmarkSplit::Train, false);
        let parsed = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::Libero, BenchmarkSplit::Train),
        )
        .unwrap();

        assert_eq!(parsed.row_count, 800);
        assert_eq!(parsed.manifest.macro9_groups.len(), 0);
        assert!(super::super::roborecover_score::validate_manifest(&parsed.manifest).is_ok());
        assert!(parsed.group_counts.contains_key(&StageTypeGroup {
            stage: RecoveryStage::Approach,
            deviation: DeviationType::WrongPose,
        }));
    }

    #[test]
    fn rejects_wrong_source_revision_or_split_blob_identity() {
        let csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        let mut ctx = context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test);
        ctx.source_revision = "main".to_owned();

        let errors = parse_pinned_split_csv(&csv, ctx).unwrap_err();
        assert!(errors.contains(&SplitManifestIssue::WrongSourceRevision));

        let mut ctx = context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test);
        ctx.split_manifest_blob_sha = "0".repeat(40);
        let errors = parse_pinned_split_csv(&csv, ctx).unwrap_err();
        assert!(errors.contains(&SplitManifestIssue::WrongSplitManifestBlobSha));
    }

    #[test]
    fn rejects_wrong_header_and_wrong_scenario_count() {
        let errors = parse_pinned_split_csv(
            "sample_id,path\n",
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.contains(&SplitManifestIssue::HeaderMismatch));
        assert!(errors.contains(&SplitManifestIssue::WrongScenarioCount {
            expected: 200,
            actual: 0,
        }));
    }

    #[test]
    fn rejects_duplicate_ids_and_paths() {
        let mut csv = vec![EXPECTED_HEADER.to_owned()];
        let duplicate = row(
            "same-id",
            BenchmarkPlatform::RoboTwin,
            BenchmarkSplit::Test,
            "Approach",
            "Wrong Pose",
        );
        csv.push(duplicate.clone());
        csv.push(duplicate);
        for index in 0..198 {
            csv.push(row(
                &format!("sample-{index:03}"),
                BenchmarkPlatform::RoboTwin,
                BenchmarkSplit::Test,
                "Approach",
                "Wrong Pose",
            ));
        }

        let errors = parse_pinned_split_csv(
            &(csv.join("\n") + "\n"),
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::DuplicateSampleId { .. }
        )));
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::DuplicatePath { .. }
        )));
    }

    #[test]
    fn rejects_path_traversal_and_sample_path_aliasing() {
        let mut csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        csv = csv.replacen(
            "robotwin/test/sample-000.json",
            "robotwin/test/../sample-000.json",
            1,
        );
        let errors = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::InvalidPath { .. }
        )));
    }

    #[test]
    fn rejects_malformed_hashes_nonpositive_actions_and_prefix_mismatches() {
        let mut csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        let valid_hash = "c".repeat(64);
        csv = csv.replacen(valid_hash.as_str(), "not-a-sha", 1);
        let errors = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.contains(&SplitManifestIssue::InvalidSha256 {
            line: 2,
            field: "sha256",
        }));

        let mut csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        csv = csv.replacen(",12,12,", ",0,12,", 1);
        let errors = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.contains(&SplitManifestIssue::InvalidActionCount {
            line: 2,
            field: "actions",
        }));

        let mut csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        csv = csv.replacen(",12,12,", ",12,11,", 1);
        let errors = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.contains(&SplitManifestIssue::ActionPrefixMismatch { line: 2 }));
    }

    #[test]
    fn rejects_missing_or_unexpected_test_macro9_groups() {
        let csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, false);
        let errors = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::MissingTestMacro9Group(_)
        )));
    }

    #[test]
    fn rejects_unknown_stage_or_deviation_without_guessing() {
        let mut csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        csv = csv.replacen(",Approach,Wrong Pose", ",Unknown,Wrong Pose", 1);
        let errors = parse_pinned_split_csv(
            &csv,
            context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test),
        )
        .unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::UnknownStage { .. }
        )));
    }

    #[test]
    fn exact_test_stage_type_counts_are_platform_specific_and_frozen() {
        for platform in [BenchmarkPlatform::Libero, BenchmarkPlatform::RoboTwin] {
            let csv = synthetic_csv(platform, BenchmarkSplit::Test, true);
            let parsed = parse_pinned_split_csv(&csv, context(platform, BenchmarkSplit::Test))
                .unwrap();
            assert_eq!(parsed.row_count, 200);
            for group in &TEST_MACRO9 {
                assert_eq!(
                    parsed.group_counts.get(group).copied(),
                    expected_test_group_count(platform, *group)
                );
            }
        }
    }

    #[test]
    fn rejects_test_macro9_group_count_drift_even_when_all_nine_groups_exist() {
        let platform = BenchmarkPlatform::RoboTwin;
        let mut csv = synthetic_csv(platform, BenchmarkSplit::Test, true);
        // Move one record from Approach/Wrong Pose to Approach/Wrong Target:
        // all nine groups remain present, but two frozen group counts drift.
        let first = row(
            "sample-000",
            platform,
            BenchmarkSplit::Test,
            "Approach",
            "Wrong Target",
        );
        let old = row(
            "sample-000",
            platform,
            BenchmarkSplit::Test,
            "Approach",
            "Wrong Pose",
        );
        csv = csv.replacen(old.as_str(), first.as_str(), 1);

        let errors = parse_pinned_split_csv(&csv, context(platform, BenchmarkSplit::Test))
            .unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::TestGroupCountMismatch { .. }
        )));
    }


    #[test]
    fn training_manifest_retains_the_tenth_training_only_group() {
        let platform = BenchmarkPlatform::Libero;
        let split = BenchmarkSplit::Train;
        let groups = [
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::WrongTarget },
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::StageRollback },
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::SceneChange },
            StageTypeGroup { stage: RecoveryStage::Contact, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Move, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Move, deviation: DeviationType::WrongTarget },
            StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::WrongTarget },
            StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::SceneChange },
        ];
        let mut rows = vec![EXPECTED_HEADER.to_owned()];
        for index in 0..800 {
            let group = groups[index % groups.len()];
            let stage = match group.stage {
                RecoveryStage::Approach => "Approach",
                RecoveryStage::Contact => "Contact",
                RecoveryStage::Move => "Move",
                RecoveryStage::Place => "Place",
            };
            let deviation = match group.deviation {
                DeviationType::WrongTarget => "Wrong Target",
                DeviationType::WrongPose => "Wrong Pose",
                DeviationType::StageRollback => "Stage Rollback",
                DeviationType::SceneChange => "Scene Change",
            };
            rows.push(row(
                &format!("train-{index:03}"),
                platform,
                split,
                stage,
                deviation,
            ));
        }
        let parsed = parse_pinned_split_csv(
            &(rows.join("\n") + "\n"),
            context(platform, split),
        ).unwrap();
        assert_eq!(parsed.manifest.macro9_groups.len(), 0);
        assert!(parsed.group_counts.contains_key(&StageTypeGroup {
            stage: RecoveryStage::Move,
            deviation: DeviationType::WrongTarget,
        }));
        assert!(super::super::roborecover_score::validate_manifest(&parsed.manifest).is_ok());
    }


    #[test]
    fn invalid_policy_or_evaluator_digest_cannot_produce_a_parsed_manifest() {
        let csv = synthetic_csv(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test, true);
        let mut ctx = context(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Test);
        ctx.policy_digest = "policy".to_owned();

        let errors = parse_pinned_split_csv(&csv, ctx).unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            SplitManifestIssue::InvalidScoreManifest(manifest_issues)
                if manifest_issues.contains(&ManifestIssue::InvalidPolicyDigest)
        )));
    }

}
