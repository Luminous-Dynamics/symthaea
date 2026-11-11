//! Fail-closed accounting for train-derived RoboRecover challenge discovery.
//!
//! This module does not generate scenarios or execute a simulator. It audits
//! candidate receipts from an external discovery runner, counts only replay-
//! verified and task-valid candidates, removes duplicate identities/content,
//! and reports recoverable/unrecoverable/unknown simulator outcomes separately.
//! It never labels simulator recovery as physical robot qualification.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    InterventionKind, WorldStateContinuity,
    roborecover_manifest::{
        pinned_split_blob_sha, PinnedTrainingParentFrame, TrainingParentScenario,
        ROBORECOVER_SOURCE_REPOSITORY, ROBORECOVER_SOURCE_REVISION,
    },
    roborecover_score::{BenchmarkPlatform, BenchmarkSplit, DeviationType, RecoveryStage},
};

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryCampaignManifest {
    pub campaign_id: String,
    pub platform: BenchmarkPlatform,
    /// Challenge generation is restricted to the official training split.
    pub source_split: BenchmarkSplit,
    /// Opaque parent frame emitted by the pinned split CSV importer.
    pub pinned_training_parent_frame: PinnedTrainingParentFrame,
    pub source_repository: String,
    pub source_revision: String,
    pub train_split_manifest_blob_sha: String,
    pub simulator_digest: String,
    pub evaluator_digest: String,
    pub generator_digest: String,
    pub observation_action_schema_digest: String,
    pub target_policy_id: String,
    pub target_policy_digest: String,
    /// Aggregate measured simulator execution time for this frozen campaign.
    pub simulator_hours: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayDisposition {
    Verified,
    Mismatch,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskValidity {
    Valid,
    Invalid,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyOutcome {
    Completed,
    Failed,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyObservation {
    Compliant,
    Violation,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlternativePolicyAttempt {
    pub platform: BenchmarkPlatform,
    pub policy_id: String,
    pub policy_digest: String,
    pub observation_action_schema_digest: String,
    pub outcome: PolicyOutcome,
    pub safety: SafetyObservation,
    pub intervention: InterventionKind,
    pub world_state: WorldStateContinuity,
    pub executed_action_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryCandidate {
    pub candidate_id: String,
    pub campaign_id: String,
    pub platform: BenchmarkPlatform,
    pub source_split: BenchmarkSplit,
    pub source_repository: String,
    pub source_revision: String,
    pub train_split_manifest_blob_sha: String,
    pub simulator_digest: String,
    pub evaluator_digest: String,
    pub generator_digest: String,
    pub observation_action_schema_digest: String,
    pub target_policy_id: String,
    pub target_policy_digest: String,
    pub parent_sample_id: String,
    pub parent_scenario_sha256: String,
    pub action_prefix_sha256: String,
    pub perturbation_sha256: String,
    pub candidate_scenario_sha256: String,
    pub task_name: String,
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    /// Evaluator-side cluster assignment. None means not yet adjudicated.
    pub failure_cluster_id: Option<String>,
    pub replay: ReplayDisposition,
    pub task_validity: TaskValidity,
    pub target_policy_outcome: PolicyOutcome,
    pub target_policy_safety: SafetyObservation,
    pub alternative_attempts: Vec<AlternativePolicyAttempt>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiscoveryManifestIssue {
    BlankCampaignId,
    NonTrainingSourceSplit,
    WrongRepository,
    WrongSourceRevision,
    WrongTrainingSplitBlob,
    InvalidDigest(&'static str),
    BlankTargetPolicyId,
    InvalidSimulatorHours,
    InvalidParentFrame,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RecoveryDiscoveryClass {
    RecoverableByNonRewindAlternative,
    UnrecoverableWithinEvaluatedPortfolio,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryCampaignReport {
    pub campaign_id: String,
    pub platform: BenchmarkPlatform,
    pub simulator_hours: f64,
    pub attempted_candidate_rows: usize,
    pub unique_candidate_ids: usize,
    pub duplicate_candidate_id_rows: usize,
    pub duplicate_scenario_content_rows: usize,
    pub invalid_identity_rows: usize,
    pub invalid_hash_rows: usize,
    pub replay_mismatch_rows: usize,
    pub replay_unknown_rows: usize,
    pub task_invalid_rows: usize,
    pub task_unknown_rows: usize,
    pub valid_candidates: usize,
    pub valid_hard_failure_candidates: usize,
    pub unique_valid_failure_discoveries: usize,
    pub unique_failure_clusters: usize,
    pub unclustered_failure_discoveries: usize,
    pub alternative_recoverable_failure_scenarios: usize,
    pub portfolio_unrecoverable_failure_scenarios: usize,
    pub unknown_recovery_scenarios: usize,
    pub simulator_only_alternative_completions: usize,
    pub invalid_alternative_attempts: usize,
    pub valid_failure_discoveries_per_simulator_hour: f64,
    pub unique_failure_clusters_per_simulator_hour: f64,
    pub recoverability: BTreeMap<RecoveryDiscoveryClass, usize>,
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
        })
}

pub fn validate_campaign_manifest(
    manifest: &DiscoveryCampaignManifest,
) -> Result<(), Vec<DiscoveryManifestIssue>> {
    let mut issues = Vec::new();
    if manifest.campaign_id.trim().is_empty() {
        issues.push(DiscoveryManifestIssue::BlankCampaignId);
    }
    if manifest.source_split != BenchmarkSplit::Train {
        issues.push(DiscoveryManifestIssue::NonTrainingSourceSplit);
    }
    if manifest.source_repository != ROBORECOVER_SOURCE_REPOSITORY {
        issues.push(DiscoveryManifestIssue::WrongRepository);
    }
    if manifest.source_revision != ROBORECOVER_SOURCE_REVISION {
        issues.push(DiscoveryManifestIssue::WrongSourceRevision);
    }
    if manifest.train_split_manifest_blob_sha
        != pinned_split_blob_sha(manifest.platform, BenchmarkSplit::Train)
    {
        issues.push(DiscoveryManifestIssue::WrongTrainingSplitBlob);
    }

    for (name, digest) in [
        ("simulator", manifest.simulator_digest.as_str()),
        ("evaluator", manifest.evaluator_digest.as_str()),
        ("generator", manifest.generator_digest.as_str()),
        ("observation_action_schema", manifest.observation_action_schema_digest.as_str()),
        ("target_policy", manifest.target_policy_digest.as_str()),
    ] {
        if !valid_hex(digest, 64) {
            issues.push(DiscoveryManifestIssue::InvalidDigest(name));
        }
    }

    if manifest.target_policy_id.trim().is_empty() {
        issues.push(DiscoveryManifestIssue::BlankTargetPolicyId);
    }
    if !manifest.simulator_hours.is_finite() || manifest.simulator_hours <= 0.0 {
        issues.push(DiscoveryManifestIssue::InvalidSimulatorHours);
    }
    if manifest.pinned_training_parent_frame.len() != 800
        || !manifest.pinned_training_parent_frame.matches_source(
            &manifest.source_repository,
            &manifest.source_revision,
            &manifest.train_split_manifest_blob_sha,
        )
    {
        issues.push(DiscoveryManifestIssue::InvalidParentFrame);
    }

    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

fn candidate_identity_matches(
    candidate: &DiscoveryCandidate,
    manifest: &DiscoveryCampaignManifest,
) -> bool {
    let Some(parent) = manifest.pinned_training_parent_frame.get(&candidate.parent_sample_id) else {
        return false;
    };

    candidate.campaign_id == manifest.campaign_id
        && candidate.platform == manifest.platform
        && candidate.source_split == BenchmarkSplit::Train
        && candidate.source_repository == manifest.source_repository
        && candidate.source_revision == manifest.source_revision
        && candidate.train_split_manifest_blob_sha == manifest.train_split_manifest_blob_sha
        && candidate.simulator_digest == manifest.simulator_digest
        && candidate.evaluator_digest == manifest.evaluator_digest
        && candidate.generator_digest == manifest.generator_digest
        && candidate.observation_action_schema_digest == manifest.observation_action_schema_digest
        && candidate.target_policy_id == manifest.target_policy_id
        && candidate.target_policy_digest == manifest.target_policy_digest
        && parent.scenario_sha256 == candidate.parent_scenario_sha256
        && parent.task_name == candidate.task_name
        && parent.stage == candidate.stage
        && parent.deviation == candidate.deviation
}

fn candidate_hashes_valid(candidate: &DiscoveryCandidate) -> bool {
    valid_id(&candidate.candidate_id)
        && valid_id(&candidate.parent_sample_id)
        && !candidate.task_name.trim().is_empty()
        && valid_hex(&candidate.parent_scenario_sha256, 64)
        && valid_hex(&candidate.action_prefix_sha256, 64)
        && valid_hex(&candidate.perturbation_sha256, 64)
        && valid_hex(&candidate.candidate_scenario_sha256, 64)
}

fn alternative_attempts_valid(
    candidate: &DiscoveryCandidate,
    manifest: &DiscoveryCampaignManifest,
) -> bool {
    let mut policies = BTreeSet::new();
    candidate.alternative_attempts.iter().all(|attempt| {
        attempt.platform == manifest.platform
            && !attempt.policy_id.trim().is_empty()
            && valid_hex(&attempt.policy_digest, 64)
            && attempt.policy_digest != manifest.target_policy_digest
            && attempt.observation_action_schema_digest
                == manifest.observation_action_schema_digest
            && policies.insert(attempt.policy_digest.clone())
    })
}

fn recovery_class(candidate: &DiscoveryCandidate) -> (RecoveryDiscoveryClass, usize) {
    let mut simulator_only = 0;
    let mut any_unknown = false;
    let mut non_rewind_safe_recovery = false;

    for attempt in &candidate.alternative_attempts {
        if attempt.outcome == PolicyOutcome::Completed
            && attempt.intervention == InterventionKind::SimulatorStateRewind
        {
            simulator_only += 1;
        }

        if attempt.outcome == PolicyOutcome::Unknown
            || attempt.safety == SafetyObservation::Unknown
            || attempt.world_state == WorldStateContinuity::Unknown
        {
            any_unknown = true;
            continue;
        }

        if attempt.outcome == PolicyOutcome::Completed
            && attempt.safety == SafetyObservation::Compliant
            && attempt.world_state == WorldStateContinuity::Preserved
            && attempt.intervention.can_be_physically_realized()
            && attempt.executed_action_count > 0
        {
            non_rewind_safe_recovery = true;
        }
    }

    if non_rewind_safe_recovery {
        (RecoveryDiscoveryClass::RecoverableByNonRewindAlternative, simulator_only)
    } else if any_unknown || candidate.alternative_attempts.is_empty() {
        (RecoveryDiscoveryClass::Unknown, simulator_only)
    } else {
        (RecoveryDiscoveryClass::UnrecoverableWithinEvaluatedPortfolio, simulator_only)
    }
}

/// Score one frozen failure-discovery campaign.
///
/// Duplicate candidate IDs and duplicate generated scenario hashes are both
/// excluded as whole duplicate groups. A hard failure means the target policy
/// failed or violated the declared safety observation on a replay-verified,
/// independently task-valid scenario. This is simulator evidence only.
pub fn score_discovery_campaign(
    manifest: &DiscoveryCampaignManifest,
    candidates: &[DiscoveryCandidate],
) -> Result<DiscoveryCampaignReport, Vec<DiscoveryManifestIssue>> {
    validate_campaign_manifest(manifest)?;

    let mut id_counts = BTreeMap::<String, usize>::new();
    let mut hash_counts = BTreeMap::<String, usize>::new();
    for candidate in candidates {
        *id_counts.entry(candidate.candidate_id.clone()).or_insert(0) += 1;
        *hash_counts
            .entry(candidate.candidate_scenario_sha256.clone())
            .or_insert(0) += 1;
    }

    let duplicate_ids = id_counts.iter()
        .filter_map(|(id, count)| (*count > 1).then(|| id.clone()))
        .collect::<BTreeSet<_>>();
    let duplicate_hashes = hash_counts.iter()
        .filter_map(|(hash, count)| (*count > 1).then(|| hash.clone()))
        .collect::<BTreeSet<_>>();

    let duplicate_candidate_id_rows = id_counts.values()
        .filter(|count| **count > 1)
        .map(|count| *count)
        .sum();
    let duplicate_scenario_content_rows = hash_counts.values()
        .filter(|count| **count > 1)
        .map(|count| *count)
        .sum();

    let mut report = DiscoveryCampaignReport {
        campaign_id: manifest.campaign_id.clone(),
        platform: manifest.platform,
        simulator_hours: manifest.simulator_hours,
        attempted_candidate_rows: candidates.len(),
        unique_candidate_ids: id_counts.len(),
        duplicate_candidate_id_rows,
        duplicate_scenario_content_rows,
        invalid_identity_rows: 0,
        invalid_hash_rows: 0,
        replay_mismatch_rows: 0,
        replay_unknown_rows: 0,
        task_invalid_rows: 0,
        task_unknown_rows: 0,
        valid_candidates: 0,
        valid_hard_failure_candidates: 0,
        unique_valid_failure_discoveries: 0,
        unique_failure_clusters: 0,
        unclustered_failure_discoveries: 0,
        alternative_recoverable_failure_scenarios: 0,
        portfolio_unrecoverable_failure_scenarios: 0,
        unknown_recovery_scenarios: 0,
        simulator_only_alternative_completions: 0,
        invalid_alternative_attempts: 0,
        valid_failure_discoveries_per_simulator_hour: 0.0,
        unique_failure_clusters_per_simulator_hour: 0.0,
        recoverability: BTreeMap::new(),
    };

    let mut unique_failure_hashes = BTreeSet::new();
    let mut failure_clusters = BTreeSet::new();

    for candidate in candidates {
        if duplicate_ids.contains(&candidate.candidate_id)
            || duplicate_hashes.contains(&candidate.candidate_scenario_sha256)
        {
            continue;
        }

        if !candidate_identity_matches(candidate, manifest) {
            report.invalid_identity_rows += 1;
            continue;
        }
        if !candidate_hashes_valid(candidate) {
            report.invalid_hash_rows += 1;
            continue;
        }
        if !alternative_attempts_valid(candidate, manifest) {
            report.invalid_alternative_attempts += 1;
            continue;
        }

        match candidate.replay {
            ReplayDisposition::Mismatch => {
                report.replay_mismatch_rows += 1;
                continue;
            }
            ReplayDisposition::Unknown => {
                report.replay_unknown_rows += 1;
                continue;
            }
            ReplayDisposition::Verified => {}
        }

        match candidate.task_validity {
            TaskValidity::Invalid => {
                report.task_invalid_rows += 1;
                continue;
            }
            TaskValidity::Unknown => {
                report.task_unknown_rows += 1;
                continue;
            }
            TaskValidity::Valid => {}
        }

        report.valid_candidates += 1;
        let hard_failure = candidate.target_policy_outcome == PolicyOutcome::Failed
            || candidate.target_policy_safety == SafetyObservation::Violation;

        if !hard_failure {
            continue;
        }

        report.valid_hard_failure_candidates += 1;
        if unique_failure_hashes.insert(candidate.candidate_scenario_sha256.clone()) {
            report.unique_valid_failure_discoveries += 1;
        }

        if let Some(cluster_id) = &candidate.failure_cluster_id {
            if !cluster_id.trim().is_empty() {
                failure_clusters.insert(cluster_id.clone());
            } else {
                report.unclustered_failure_discoveries += 1;
            }
        } else {
            report.unclustered_failure_discoveries += 1;
        }

        let (class, simulator_only) = recovery_class(candidate);
        report.simulator_only_alternative_completions += simulator_only;
        *report.recoverability.entry(class).or_insert(0) += 1;
        match class {
            RecoveryDiscoveryClass::RecoverableByNonRewindAlternative => {
                report.alternative_recoverable_failure_scenarios += 1;
            }
            RecoveryDiscoveryClass::UnrecoverableWithinEvaluatedPortfolio => {
                report.portfolio_unrecoverable_failure_scenarios += 1;
            }
            RecoveryDiscoveryClass::Unknown => report.unknown_recovery_scenarios += 1,
        }
    }

    report.unique_failure_clusters = failure_clusters.len();
    report.valid_failure_discoveries_per_simulator_hour =
        report.unique_valid_failure_discoveries as f64 / manifest.simulator_hours;
    report.unique_failure_clusters_per_simulator_hour =
        report.unique_failure_clusters as f64 / manifest.simulator_hours;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parent_entries() -> BTreeMap<String, TrainingParentScenario> {
        let mut entries = BTreeMap::new();
        entries.insert(
            "parent-01".to_owned(),
            TrainingParentScenario {
                scenario_sha256: "1".repeat(64),
                task_name: "move_can_pot".to_owned(),
                stage: RecoveryStage::Approach,
                deviation: DeviationType::WrongPose,
            },
        );
        for index in 0..799 {
            entries.insert(
                format!("sample-{index:03}"),
                TrainingParentScenario {
                    scenario_sha256: format!("{index:064x}"),
                    task_name: "move_can_pot".to_owned(),
                    stage: RecoveryStage::Approach,
                    deviation: DeviationType::WrongPose,
                },
            );
        }
        entries
    }

    fn manifest() -> DiscoveryCampaignManifest {
        let source_repository = ROBORECOVER_SOURCE_REPOSITORY.to_owned();
        let source_revision = ROBORECOVER_SOURCE_REVISION.to_owned();
        let train_split_manifest_blob_sha =
            pinned_split_blob_sha(BenchmarkPlatform::RoboTwin, BenchmarkSplit::Train).to_owned();
        let pinned_training_parent_frame = PinnedTrainingParentFrame::fixture(
            &source_repository,
            &source_revision,
            &train_split_manifest_blob_sha,
            parent_entries(),
        );
        DiscoveryCampaignManifest {
            campaign_id: "campaign-01".to_owned(),
            platform: BenchmarkPlatform::RoboTwin,
            source_split: BenchmarkSplit::Train,
            pinned_training_parent_frame,
            source_repository,
            source_revision,
            train_split_manifest_blob_sha,
            simulator_digest: "a".repeat(64),
            evaluator_digest: "b".repeat(64),
            generator_digest: "c".repeat(64),
            observation_action_schema_digest: "d".repeat(64),
            target_policy_id: "target-policy".to_owned(),
            target_policy_digest: "e".repeat(64),
            simulator_hours: 2.0,
        }
    }

    fn candidate(id: &str, scenario_hash: &str) -> DiscoveryCandidate {
        let m = manifest();
        DiscoveryCandidate {
            candidate_id: id.to_owned(),
            campaign_id: m.campaign_id,
            platform: m.platform,
            source_split: m.source_split,
            source_repository: m.source_repository,
            source_revision: m.source_revision,
            train_split_manifest_blob_sha: m.train_split_manifest_blob_sha,
            simulator_digest: m.simulator_digest,
            evaluator_digest: m.evaluator_digest,
            generator_digest: m.generator_digest,
            observation_action_schema_digest: m.observation_action_schema_digest.clone(),
            target_policy_id: m.target_policy_id,
            target_policy_digest: m.target_policy_digest,
            parent_sample_id: "parent-01".to_owned(),
            parent_scenario_sha256: "1".repeat(64),
            action_prefix_sha256: "2".repeat(64),
            perturbation_sha256: "3".repeat(64),
            candidate_scenario_sha256: scenario_hash.to_owned(),
            task_name: "move_can_pot".to_owned(),
            stage: RecoveryStage::Approach,
            deviation: DeviationType::WrongPose,
            failure_cluster_id: Some("cluster-01".to_owned()),
            replay: ReplayDisposition::Verified,
            task_validity: TaskValidity::Valid,
            target_policy_outcome: PolicyOutcome::Failed,
            target_policy_safety: SafetyObservation::Compliant,
            alternative_attempts: vec![AlternativePolicyAttempt {
                platform: m.platform,
                policy_id: "alternative-policy".to_owned(),
                policy_digest: "f".repeat(64),
                observation_action_schema_digest: m.observation_action_schema_digest,
                outcome: PolicyOutcome::Completed,
                safety: SafetyObservation::Compliant,
                intervention: InterventionKind::CompensatingAction,
                world_state: WorldStateContinuity::Preserved,
                executed_action_count: 4,
            }],
        }
    }

    #[test]
    fn rejects_test_split_as_a_source_for_adaptive_challenge_discovery() {
        let mut m = manifest();
        m.source_split = BenchmarkSplit::Test;
        assert!(validate_campaign_manifest(&m)
            .unwrap_err()
            .contains(&DiscoveryManifestIssue::NonTrainingSourceSplit));
    }

    #[test]
    fn counts_only_replay_verified_task_valid_hard_failures() {
        let m = manifest();
        let mut valid = candidate("candidate-a", &"4".repeat(64));
        let mut replay_bad = candidate("candidate-b", &"5".repeat(64));
        replay_bad.replay = ReplayDisposition::Mismatch;
        let mut task_bad = candidate("candidate-c", &"6".repeat(64));
        task_bad.task_validity = TaskValidity::Invalid;
        let mut target_succeeds = candidate("candidate-d", &"7".repeat(64));
        target_succeeds.target_policy_outcome = PolicyOutcome::Completed;

        let report = score_discovery_campaign(
            &m,
            &[valid.clone(), replay_bad, task_bad, target_succeeds],
        ).unwrap();

        assert_eq!(report.attempted_candidate_rows, 4);
        assert_eq!(report.valid_candidates, 2);
        assert_eq!(report.valid_hard_failure_candidates, 1);
        assert_eq!(report.unique_valid_failure_discoveries, 1);
        assert_eq!(report.replay_mismatch_rows, 1);
        assert_eq!(report.task_invalid_rows, 1);
    }

    #[test]
    fn duplicate_scenario_content_hashes_are_excluded_as_a_group() {
        let m = manifest();
        let first = candidate("candidate-a", &"4".repeat(64));
        let mut second = candidate("candidate-b", &"4".repeat(64));
        second.failure_cluster_id = Some("cluster-02".to_owned());

        let report = score_discovery_campaign(&m, &[first, second]).unwrap();
        assert_eq!(report.duplicate_scenario_content_rows, 2);
        assert_eq!(report.unique_valid_failure_discoveries, 0);
        assert_eq!(report.valid_candidates, 0);
    }

    #[test]
    fn duplicate_candidate_ids_are_excluded_as_a_group() {
        let m = manifest();
        let first = candidate("candidate-a", &"4".repeat(64));
        let second = candidate("candidate-a", &"5".repeat(64));

        let report = score_discovery_campaign(&m, &[first, second]).unwrap();
        assert_eq!(report.duplicate_candidate_id_rows, 2);
        assert_eq!(report.unique_valid_failure_discoveries, 0);
    }

    #[test]
    fn valid_failure_clusters_are_counted_once_per_cluster() {
        let m = manifest();
        let first = candidate("candidate-a", &"4".repeat(64));
        let mut second = candidate("candidate-b", &"5".repeat(64));
        second.parent_sample_id = "parent-02".to_owned();
        second.parent_scenario_sha256 = "9".repeat(64);
        second.perturbation_sha256 = "8".repeat(64);
        let report = score_discovery_campaign(&m, &[first, second]).unwrap();

        assert_eq!(report.unique_valid_failure_discoveries, 2);
        assert_eq!(report.unique_failure_clusters, 1);
        assert_eq!(report.valid_failure_discoveries_per_simulator_hour, 1.0);
        assert_eq!(report.unique_failure_clusters_per_simulator_hour, 0.5);
    }

    #[test]
    fn unknown_replay_and_invalid_task_do_not_become_discoveries() {
        let m = manifest();
        let mut replay_unknown = candidate("candidate-a", &"4".repeat(64));
        replay_unknown.replay = ReplayDisposition::Unknown;
        let mut task_unknown = candidate("candidate-b", &"5".repeat(64));
        task_unknown.task_validity = TaskValidity::Unknown;

        let report = score_discovery_campaign(&m, &[replay_unknown, task_unknown]).unwrap();
        assert_eq!(report.unique_valid_failure_discoveries, 0);
        assert_eq!(report.replay_unknown_rows, 1);
        assert_eq!(report.task_unknown_rows, 1);
    }

    #[test]
    fn recovery_class_separates_nonrewind_safe_alternative_rewind_and_unknown() {
        let mut safe = candidate("candidate-a", &"4".repeat(64));
        let mut rewind = candidate("candidate-b", &"5".repeat(64));
        rewind.alternative_attempts[0].intervention = InterventionKind::SimulatorStateRewind;
        rewind.alternative_attempts[0].world_state = WorldStateContinuity::SimulatorRewound;
        let mut unknown = candidate("candidate-c", &"6".repeat(64));
        unknown.alternative_attempts[0].outcome = PolicyOutcome::Unknown;

        let report = score_discovery_campaign(&manifest(), &[safe, rewind, unknown]).unwrap();
        assert_eq!(report.alternative_recoverable_failure_scenarios, 1);
        assert_eq!(report.portfolio_unrecoverable_failure_scenarios, 1);
        assert_eq!(report.unknown_recovery_scenarios, 1);
        assert_eq!(report.simulator_only_alternative_completions, 1);
    }

    #[test]
    fn rejects_duplicate_or_incompatible_alternative_policy_attempts() {
        let m = manifest();
        let mut duplicate = candidate("candidate-a", &"4".repeat(64));
        duplicate.alternative_attempts.push(duplicate.alternative_attempts[0].clone());

        let mut incompatible = candidate("candidate-b", &"5".repeat(64));
        incompatible.alternative_attempts[0].observation_action_schema_digest = "0".repeat(64);

        let report = score_discovery_campaign(&m, &[duplicate, incompatible]).unwrap();
        assert_eq!(report.invalid_alternative_attempts, 2);
        assert_eq!(report.valid_candidates, 0);
    }

    #[test]
    fn provenance_and_digest_mismatches_are_not_counted() {
        let m = manifest();
        let mut wrong_generator = candidate("candidate-a", &"4".repeat(64));
        wrong_generator.generator_digest = "0".repeat(64);
        let mut malformed_hash = candidate("candidate-b", &"5".repeat(64));
        malformed_hash.perturbation_sha256 = "bad".to_owned();

        let report = score_discovery_campaign(&m, &[wrong_generator, malformed_hash]).unwrap();
        assert_eq!(report.invalid_identity_rows, 1);
        assert_eq!(report.invalid_hash_rows, 1);
        assert_eq!(report.unique_valid_failure_discoveries, 0);
    }

    #[test]
    fn rejects_parent_not_in_frozen_training_scenario_frame() {
        let m = manifest();
        let mut candidate = candidate("candidate-a", &"4".repeat(64));
        candidate.parent_sample_id = "heldout-test-sample".to_owned();

        let report = score_discovery_campaign(&m, &[candidate]).unwrap();
        assert_eq!(report.invalid_identity_rows, 1);
        assert_eq!(report.valid_candidates, 0);
        assert_eq!(report.unique_valid_failure_discoveries, 0);
    }

    #[test]
    fn rejects_parent_hash_task_or_stage_type_mismatch() {
        let m = manifest();
        let mut bad_hash = candidate("candidate-a", &"4".repeat(64));
        bad_hash.parent_scenario_sha256 = "9".repeat(64);

        let mut bad_task = candidate("candidate-b", &"5".repeat(64));
        bad_task.task_name = "another_task".to_owned();

        let mut bad_stage = candidate("candidate-c", &"6".repeat(64));
        bad_stage.deviation = DeviationType::WrongTarget;

        let report = score_discovery_campaign(&m, &[bad_hash, bad_task, bad_stage]).unwrap();
        assert_eq!(report.invalid_identity_rows, 3);
        assert_eq!(report.valid_candidates, 0);
    }

    #[test]
    fn rejects_incomplete_or_wrong_source_training_parent_frame() {
        let mut entries = parent_entries();
        entries.remove("parent-01");
        let mut m = manifest();
        m.pinned_training_parent_frame = PinnedTrainingParentFrame::fixture(
            &m.source_repository,
            &m.source_revision,
            &m.train_split_manifest_blob_sha,
            entries,
        );
        assert!(validate_campaign_manifest(&m)
            .unwrap_err()
            .contains(&DiscoveryManifestIssue::InvalidParentFrame));

        let mut m = manifest();
        m.pinned_training_parent_frame = PinnedTrainingParentFrame::fixture(
            &m.source_repository,
            &m.source_revision,
            &"0".repeat(40),
            parent_entries(),
        );
        assert!(validate_campaign_manifest(&m)
            .unwrap_err()
            .contains(&DiscoveryManifestIssue::InvalidParentFrame));
    }

}
