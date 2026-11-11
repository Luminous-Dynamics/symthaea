//! RoboRecover recovery scoring with fail-closed trial admission.
//!
//! Policy-recovery RSR is scenario-equal: compute policy-attributable success
//! per scenario, then average scenarios equally. Macro-9 RSR averages the nine
//! explicitly declared occupied Stage–Type groups equally. Replay-only task
//! completion is reported separately and never credited to the policy. Missing,
//! duplicate, unplanned, identity-invalid, or metadata-mismatched trial rows
//! never become successes; planned-but-unobserved trials stay in the denominator.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    audit_trials, classify_trial, AuditReport, TerminalOutcome,
    TrialDisposition, TrialEvidence,
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BenchmarkPlatform {
    RoboTwin,
    Libero,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BenchmarkSplit {
    Train,
    Test,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    /// Reconstruct the off-nominal state, then hand control to the policy.
    ReplayThenInfer,
    /// Evaluate the clean initial state without a recovery handoff.
    SeedOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioSource {
    Natural,
    ActionPerturbed,
    HumanConstructed,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RecoveryStage {
    Approach,
    Contact,
    Move,
    Place,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DeviationType {
    WrongTarget,
    WrongPose,
    StageRollback,
    SceneChange,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StageTypeGroup {
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedScenario {
    pub scenario_id: String,
    /// Exact relative path in the pinned upstream scenario archive.
    pub source_path: String,
    /// SHA-256 of the reconstructed/off-nominal scenario JSON.
    pub scenario_content_sha256: String,
    /// SHA-256 of the original trajectory JSON from which the scenario derives.
    pub original_trajectory_sha256: String,
    pub task_name: String,
    pub actions: u32,
    pub replay_prefix_steps: u32,
    /// Evaluator-side construction provenance; never a policy/router feature.
    pub source: ScenarioSource,
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    /// Expected independent rollout IDs for this scenario.
    pub expected_trial_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoboRecoverManifest {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    /// Must be ReplayThenInfer for confirmatory recovery scoring.
    pub execution_mode: ExecutionMode,
    /// Canonical upstream repository and exact source revision.
    pub source_repository: String,
    pub source_revision: String,
    /// Git blob SHA of the exact upstream splits/<platform>_<split>.csv file.
    pub split_manifest_blob_sha: String,
    /// SHA-256 of the exact policy checkpoint/configuration used by this arm.
    pub policy_digest: String,
    /// SHA-256 of the exact policy-specific evaluator adapter.
    pub evaluator_digest: String,
    /// One score is one policy/intervention arm; never pool arm IDs.
    pub arm_id: String,
    pub scenarios: Vec<PlannedScenario>,
    /// The benchmark's nine occupied Stage–Type groups, frozen from source.
    /// Do not infer or pad these from whichever scenarios happened to execute.
    pub macro9_groups: Vec<StageTypeGroup>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyCompletion {
    /// The evaluator records success after policy inference had control.
    PolicyCompleted,
    /// The replay prefix or post-replay wait completed the environment before
    /// policy inference could take responsibility for recovery.
    ReplayPreempted,
    /// A known evaluator outcome says the policy did not complete the task.
    NotCompleted,
    /// The source status/fields do not establish whether policy recovery occurred.
    Unknown,
    /// Source status and success/inference fields contradict one another.
    Contradictory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoboRecoverTrial {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    pub execution_mode: ExecutionMode,
    pub source_repository: String,
    pub source_path: String,
    pub source_revision: String,
    pub split_manifest_blob_sha: String,
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    pub scenario_content_sha256: String,
    pub original_trajectory_sha256: String,
    pub task_name: String,
    pub actions: u32,
    pub replay_prefix_steps: u32,
    /// Evaluator-side construction provenance; never a policy/router feature.
    pub source: ScenarioSource,
    /// Exact status string emitted by the pinned source adapter.
    pub source_status: String,
    /// Number of actions actually executed after policy handoff.
    pub inference_steps: u32,
    /// Raw RoboTwin `eval_success_at_end`; LIBERO's adapter has no equivalent
    /// field and supplies None (its success is represented by status).
    pub source_terminal_success: Option<bool>,
    pub evidence: TrialEvidence,
}

/// Normalize source-specific fields without conflating environment completion
/// with a task completed by the evaluated policy.
pub fn classify_policy_completion(trial: &RoboRecoverTrial) -> PolicyCompletion {
    if trial.source_status == "replay_already_done" {
        return match trial.platform {
            BenchmarkPlatform::RoboTwin => match (
                trial.inference_steps,
                trial.source_terminal_success,
            ) {
                (0, Some(true)) => PolicyCompletion::ReplayPreempted,
                (0, None) => PolicyCompletion::Unknown,
                _ => PolicyCompletion::Contradictory,
            },
            BenchmarkPlatform::Libero if trial.inference_steps == 0 => {
                PolicyCompletion::ReplayPreempted
            }
            BenchmarkPlatform::Libero => PolicyCompletion::Contradictory,
        };
    }

    if trial.platform == BenchmarkPlatform::RoboTwin {
        return match (
            trial.source_status.as_str(),
            trial.inference_steps,
            trial.source_terminal_success,
        ) {
            ("infer_ok", steps, Some(true))
                if steps > 0 && trial.evidence.terminal == TerminalOutcome::TaskCompleted =>
            {
                PolicyCompletion::PolicyCompleted
            }
            ("infer_ok", steps, Some(true)) if steps > 0 => PolicyCompletion::Contradictory,
            ("infer_ok", _, Some(false)) => PolicyCompletion::Contradictory,
            ("infer_ok", _, None) => PolicyCompletion::Unknown,
            ("infer_ok", 0, Some(true)) => PolicyCompletion::Contradictory,
            ("replay_ok", 0, Some(true)) => PolicyCompletion::ReplayPreempted,
            ("replay_ok", 0, None) => PolicyCompletion::Unknown,
            ("replay_ok", 0, Some(false)) => PolicyCompletion::NotCompleted,
            ("replay_ok", _, _) => PolicyCompletion::Contradictory,
            ("infer_failed" | "max_infer_steps_reached" | "state_mismatch" | "eval_crash", _, Some(true)) => {
                PolicyCompletion::Contradictory
            }
            ("infer_failed" | "max_infer_steps_reached" | "state_mismatch" | "eval_crash", _, _) => {
                PolicyCompletion::NotCompleted
            }
            _ => PolicyCompletion::Unknown,
        };
    }

    match trial.source_status.as_str() {
        "success"
            if trial.inference_steps > 0
                && trial.evidence.terminal == TerminalOutcome::TaskCompleted =>
        {
            PolicyCompletion::PolicyCompleted
        }
        "success" if trial.inference_steps > 0 => PolicyCompletion::Contradictory,
        "success" => PolicyCompletion::Contradictory,
        "state_mismatch" | "timeout_failure" | "env_crash" | "invalid_sample" => {
            PolicyCompletion::NotCompleted
        }
        _ => PolicyCompletion::Unknown,
    }
}

/// Whether the source evaluator reports an environment-completion flag/status.
/// This is not evidence that the policy caused completion.
pub fn source_reports_environment_completion(trial: &RoboRecoverTrial) -> Option<bool> {
    match trial.platform {
        BenchmarkPlatform::RoboTwin => trial.source_terminal_success,
        BenchmarkPlatform::Libero => match trial.source_status.as_str() {
            "success" | "replay_already_done" => Some(true),
            "state_mismatch" | "timeout_failure" | "env_crash" | "invalid_sample" => Some(false),
            _ => None,
        },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestIssue {
    NoScenarios,
    SeedOnlyIsNotRecovery,
    BlankArmId,
    BlankSourceRepository,
    BlankSourceRevision,
    BlankSplitManifestBlobSha,
    BlankPolicyDigest,
    BlankEvaluatorDigest,
    InvalidSourceRevision,
    InvalidSplitManifestBlobSha,
    InvalidPolicyDigest,
    InvalidEvaluatorDigest,
    BlankScenarioDigest(String),
    InvalidScenarioDigest(String),
    InvalidOriginalTrajectoryDigest(String),
    InvalidScenarioPath(String),
    BlankTaskName(String),
    InvalidActionPrefix(String),
    BlankScenarioId,
    ScenarioGroupNotInMacro9(StageTypeGroup),
    DuplicateScenarioId(String),
    ScenarioHasNoExpectedTrials(String),
    BlankExpectedTrialId(String),
    DuplicateExpectedTrialId(String),
    Macro9GroupCount(usize),
    Macro9OnlyAppliesToTest,
    DuplicateMacro9Group(StageTypeGroup),
    Macro9GroupHasNoScenarios(StageTypeGroup),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioScore {
    pub scenario_id: String,
    pub source_path: String,
    pub scenario_content_sha256: String,
    pub original_trajectory_sha256: String,
    pub task_name: String,
    pub actions: u32,
    pub replay_prefix_steps: u32,
    pub source: ScenarioSource,
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    pub expected_trials: usize,
    pub accepted_trial_rows: usize,
    pub source_environment_completions: usize,
    pub policy_recovery_successes: usize,
    pub physical_safe_successes: usize,
    pub source_environment_completion_rsr: f64,
    pub policy_recovery_rsr: f64,
    pub physical_safe_rsr: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupScore {
    pub group: StageTypeGroup,
    pub scenario_count: usize,
    pub source_environment_completion_rsr: f64,
    pub policy_recovery_rsr: f64,
    pub physical_safe_rsr: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoboRecoverScoreReport {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    pub execution_mode: ExecutionMode,
    pub source_repository: String,
    pub source_revision: String,
    pub split_manifest_blob_sha: String,
    pub policy_digest: String,
    pub evaluator_digest: String,
    pub arm_id: String,
    pub scenario_count: usize,
    pub expected_trial_count: usize,
    pub observed_trial_rows: usize,
    pub accepted_trial_rows: usize,
    pub metadata_mismatch_rows: usize,
    pub source_environment_completions: usize,
    pub replay_preempted_rows: usize,
    pub policy_recovery_successes: usize,
    pub policy_not_completed_rows: usize,
    pub unknown_policy_outcome_rows: usize,
    pub contradictory_policy_outcome_rows: usize,
    pub physical_safe_successes: usize,
    pub missing_expected_trial_ids: Vec<String>,
    pub duplicate_trial_ids: Vec<String>,
    /// Low-level trial-ID/identity audit. The accepted_trial_rows count above
    /// additionally applies platform, split, source, policy and evaluator matching.
    pub identity_audit: AuditReport,
    pub scenarios: Vec<ScenarioScore>,
    pub groups: Vec<GroupScore>,
    /// Scenario-equal source-reported environment completion rate. May include replay preemption.
    pub source_environment_completion_overall_rsr: f64,
    /// Equal-weight mean of source environment-completion rates across Macro-9 (test only).
    pub source_environment_completion_macro9_rsr: Option<f64>,
    /// Scenario-equal policy-attributable recovery RSR, excluding prefix replay.
    pub policy_recovery_overall_rsr: f64,
    /// Equal-weight mean across the nine occupied Stage–Type groups.
    pub policy_recovery_macro9_rsr: Option<f64>,
    /// Luminous extension: scenario-equal safe completion without state rewind.
    pub physical_safe_overall_rsr: f64,
    /// Luminous extension: equal-weight group mean of physical-safe RSR.
    pub physical_safe_macro9_rsr: Option<f64>,
}

#[derive(Default)]
struct ScenarioAccumulator {
    accepted_trial_rows: usize,
    source_environment_completions: usize,
    policy_recovery_successes: usize,
    physical_safe_successes: usize,
}

/// Validate the frozen RoboRecover scenario and expected-trial manifest.
///
/// The test split requires nine explicit, unique Macro-9 groups. The training
/// split forbids Macro-9 declarations and permits training-only Stage–Type
/// combinations. Neither split accepts clean seed-only runs as recovery.
fn is_lower_or_upper_hex_digest(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Validate source identity, split membership and scenario digests before scoring.
pub fn validate_manifest(
    manifest: &RoboRecoverManifest,
) -> Result<(), Vec<ManifestIssue>> {
    let mut issues = Vec::new();
    if manifest.scenarios.is_empty() {
        issues.push(ManifestIssue::NoScenarios);
    }
    if manifest.execution_mode != ExecutionMode::ReplayThenInfer {
        issues.push(ManifestIssue::SeedOnlyIsNotRecovery);
    }
    if manifest.arm_id.trim().is_empty() {
        issues.push(ManifestIssue::BlankArmId);
    }
    if manifest.source_repository.trim().is_empty() {
        issues.push(ManifestIssue::BlankSourceRepository);
    }
    if manifest.source_revision.trim().is_empty() {
        issues.push(ManifestIssue::BlankSourceRevision);
    } else if !is_lower_or_upper_hex_digest(&manifest.source_revision, 40) {
        issues.push(ManifestIssue::InvalidSourceRevision);
    }
    if manifest.split_manifest_blob_sha.trim().is_empty() {
        issues.push(ManifestIssue::BlankSplitManifestBlobSha);
    } else if !is_lower_or_upper_hex_digest(&manifest.split_manifest_blob_sha, 40) {
        issues.push(ManifestIssue::InvalidSplitManifestBlobSha);
    }
    if manifest.policy_digest.trim().is_empty() {
        issues.push(ManifestIssue::BlankPolicyDigest);
    } else if !is_lower_or_upper_hex_digest(&manifest.policy_digest, 64) {
        issues.push(ManifestIssue::InvalidPolicyDigest);
    }
    if manifest.evaluator_digest.trim().is_empty() {
        issues.push(ManifestIssue::BlankEvaluatorDigest);
    } else if !is_lower_or_upper_hex_digest(&manifest.evaluator_digest, 64) {
        issues.push(ManifestIssue::InvalidEvaluatorDigest);
    }

    let mut scenario_ids = BTreeSet::new();
    let mut expected_trial_ids = BTreeSet::new();
    let mut group_presence = BTreeSet::new();

    for scenario in &manifest.scenarios {
        if scenario.scenario_content_sha256.trim().is_empty() {
            issues.push(ManifestIssue::BlankScenarioDigest(
                scenario.scenario_id.clone(),
            ));
        } else if !is_lower_or_upper_hex_digest(&scenario.scenario_content_sha256, 64) {
            issues.push(ManifestIssue::InvalidScenarioDigest(
                scenario.scenario_id.clone(),
            ));
        }
        if !is_lower_or_upper_hex_digest(&scenario.original_trajectory_sha256, 64) {
            issues.push(ManifestIssue::InvalidOriginalTrajectoryDigest(
                scenario.scenario_id.clone(),
            ));
        }
        if scenario.source_path.trim().is_empty()
            || scenario.source_path.starts_with('/')
            || scenario.source_path.contains('\\')
            || scenario.source_path.split('/').any(|part| part == ".." || part == ".")
            || !scenario.source_path.ends_with(&format!("/{}.json", scenario.scenario_id))
        {
            issues.push(ManifestIssue::InvalidScenarioPath(
                scenario.scenario_id.clone(),
            ));
        }
        if scenario.task_name.trim().is_empty() {
            issues.push(ManifestIssue::BlankTaskName(scenario.scenario_id.clone()));
        }
        if scenario.actions == 0
            || scenario.replay_prefix_steps == 0
            || scenario.actions != scenario.replay_prefix_steps
        {
            issues.push(ManifestIssue::InvalidActionPrefix(
                scenario.scenario_id.clone(),
            ));
        }
        if scenario.scenario_id.trim().is_empty() {
            issues.push(ManifestIssue::BlankScenarioId);
        } else if !scenario_ids.insert(scenario.scenario_id.clone()) {
            issues.push(ManifestIssue::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }

        if scenario.expected_trial_ids.is_empty() {
            issues.push(ManifestIssue::ScenarioHasNoExpectedTrials(
                scenario.scenario_id.clone(),
            ));
        }

        group_presence.insert(StageTypeGroup {
            stage: scenario.stage,
            deviation: scenario.deviation,
        });

        for trial_id in &scenario.expected_trial_ids {
            if trial_id.trim().is_empty() {
                issues.push(ManifestIssue::BlankExpectedTrialId(
                    scenario.scenario_id.clone(),
                ));
            } else if !expected_trial_ids.insert(trial_id.clone()) {
                issues.push(ManifestIssue::DuplicateExpectedTrialId(
                    trial_id.clone(),
                ));
            }
        }
    }

    if manifest.split == BenchmarkSplit::Test && manifest.macro9_groups.len() != 9 {
        issues.push(ManifestIssue::Macro9GroupCount(
            manifest.macro9_groups.len(),
        ));
    }
    if manifest.split == BenchmarkSplit::Train && !manifest.macro9_groups.is_empty() {
        issues.push(ManifestIssue::Macro9OnlyAppliesToTest);
    }

    let mut declared_groups = BTreeSet::new();
    for group in &manifest.macro9_groups {
        if !declared_groups.insert(*group) {
            issues.push(ManifestIssue::DuplicateMacro9Group(*group));
        } else if !group_presence.contains(group) {
            issues.push(ManifestIssue::Macro9GroupHasNoScenarios(*group));
        }
    }
    if manifest.split == BenchmarkSplit::Test {
        for group in &group_presence {
            if !declared_groups.contains(group) {
                issues.push(ManifestIssue::ScenarioGroupNotInMacro9(*group));
            }
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Score one platform, split and policy/intervention arm against a frozen manifest.
///
/// The policy-recovery outcome is source-protocol aware:
/// - RoboTwin requires status=infer_ok, at least one inference action, and
///   eval_success_at_end=true.
/// - LIBERO requires status=success and at least one inference action.
/// - replay_already_done is always a replay-preempted environment completion,
///   never a policy recovery.
///
/// The raw source environment-completion signal is reported separately. The
/// physical-safe extension additionally requires preserved world continuity,
/// an executed physically realizable intervention, and upstream verified
/// compliant safety evidence. This is not a claim that source summary fields
/// alone are an official reproduction of every paper result; see the README's
/// adapter caveat.
///
/// Every planned rollout remains in its scenario denominator. Missing,
/// duplicated, unplanned, identity-invalid and metadata-mismatched rows count
/// as non-successes, not silent exclusions.
pub fn score_roborecover(
    manifest: &RoboRecoverManifest,
    trials: &[RoboRecoverTrial],
) -> Result<RoboRecoverScoreReport, Vec<ManifestIssue>> {
    validate_manifest(manifest)?;

    let expected_trial_ids = manifest
        .scenarios
        .iter()
        .flat_map(|scenario| scenario.expected_trial_ids.iter().cloned())
        .collect::<Vec<_>>();

    let evidence_rows = trials
        .iter()
        .map(|trial| trial.evidence.clone())
        .collect::<Vec<_>>();
    let audit = audit_trials(&evidence_rows, Some(&expected_trial_ids));

    let duplicate_ids = audit
        .duplicate_trial_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    let mut expected_to_scenario = BTreeMap::<String, &PlannedScenario>::new();
    let mut accumulators = BTreeMap::<String, ScenarioAccumulator>::new();
    for scenario in &manifest.scenarios {
        accumulators.insert(
            scenario.scenario_id.clone(),
            ScenarioAccumulator::default(),
        );
        for trial_id in &scenario.expected_trial_ids {
            expected_to_scenario.insert(trial_id.clone(), scenario);
        }
    }

    let mut accepted_trial_rows = 0;
    let mut metadata_mismatch_rows = 0;
    let mut source_environment_completions = 0;
    let mut replay_preempted_rows = 0;
    let mut policy_recovery_successes = 0;
    let mut policy_not_completed_rows = 0;
    let mut unknown_policy_outcome_rows = 0;
    let mut contradictory_policy_outcome_rows = 0;
    let mut physical_safe_successes = 0;

    for trial in trials {
        let trial_id = &trial.evidence.trial_id;
        if duplicate_ids.contains(trial_id) {
            continue;
        }

        let Some(scenario) = expected_to_scenario.get(trial_id) else {
            continue;
        };

        if !trial.evidence.has_complete_identity() {
            continue;
        }

        let metadata_matches = trial.platform == manifest.platform
            && trial.split == manifest.split
            && trial.execution_mode == manifest.execution_mode
            && trial.source_repository == manifest.source_repository
            && trial.source_revision == manifest.source_revision
            && trial.split_manifest_blob_sha == manifest.split_manifest_blob_sha
            && trial.evidence.policy_digest == manifest.policy_digest
            && trial.evidence.arm_id == manifest.arm_id
            && trial.stage == scenario.stage
            && trial.deviation == scenario.deviation
            && trial.source == scenario.source
            && trial.evidence.scenario_id == scenario.scenario_id
            && trial.source_path == scenario.source_path
            && trial.scenario_content_sha256 == scenario.scenario_content_sha256
            && trial.original_trajectory_sha256 == scenario.original_trajectory_sha256
            && trial.task_name == scenario.task_name
            && trial.actions == scenario.actions
            && trial.replay_prefix_steps == scenario.replay_prefix_steps
            && trial.evidence.scenario_digest == scenario.scenario_content_sha256
            && trial.evidence.evaluator_digest == manifest.evaluator_digest;

        if !metadata_matches {
            metadata_mismatch_rows += 1;
            continue;
        }

        accepted_trial_rows += 1;
        let accumulator = accumulators
            .get_mut(&scenario.scenario_id)
            .expect("validated scenario must have an accumulator");
        accumulator.accepted_trial_rows += 1;

        if source_reports_environment_completion(trial) == Some(true) {
            source_environment_completions += 1;
            accumulator.source_environment_completions += 1;
        }
        match classify_policy_completion(trial) {
            PolicyCompletion::PolicyCompleted => {
                policy_recovery_successes += 1;
                accumulator.policy_recovery_successes += 1;
            }
            PolicyCompletion::ReplayPreempted => replay_preempted_rows += 1,
            PolicyCompletion::NotCompleted => policy_not_completed_rows += 1,
            PolicyCompletion::Unknown => unknown_policy_outcome_rows += 1,
            PolicyCompletion::Contradictory => contradictory_policy_outcome_rows += 1,
        }

        if classify_policy_completion(trial) == PolicyCompletion::PolicyCompleted
            && trial.evidence.terminal == TerminalOutcome::TaskCompleted
            && classify_trial(&trial.evidence) == TrialDisposition::PhysicalSafeCompletion
        {
            physical_safe_successes += 1;
            accumulator.physical_safe_successes += 1;
        }
    }

    let mut scenario_scores = Vec::with_capacity(manifest.scenarios.len());
    for scenario in &manifest.scenarios {
        let accumulator = accumulators
            .get(&scenario.scenario_id)
            .expect("validated scenario must have an accumulator");
        let denominator = scenario.expected_trial_ids.len() as f64;

        scenario_scores.push(ScenarioScore {
            scenario_id: scenario.scenario_id.clone(),
            source_path: scenario.source_path.clone(),
            scenario_content_sha256: scenario.scenario_content_sha256.clone(),
            original_trajectory_sha256: scenario.original_trajectory_sha256.clone(),
            task_name: scenario.task_name.clone(),
            actions: scenario.actions,
            replay_prefix_steps: scenario.replay_prefix_steps,
            source: scenario.source,
            stage: scenario.stage,
            deviation: scenario.deviation,
            expected_trials: scenario.expected_trial_ids.len(),
            accepted_trial_rows: accumulator.accepted_trial_rows,
            source_environment_completions: accumulator.source_environment_completions,
            policy_recovery_successes: accumulator.policy_recovery_successes,
            physical_safe_successes: accumulator.physical_safe_successes,
            source_environment_completion_rsr: accumulator.source_environment_completions as f64 / denominator,
            policy_recovery_rsr: accumulator.policy_recovery_successes as f64 / denominator,
            physical_safe_rsr: accumulator.physical_safe_successes as f64 / denominator,
        });
    }

    // Each scenario receives equal weight in the overall metric, irrespective
    // of how many repeated trials the frozen manifest assigns to it.
    let scenario_count = scenario_scores.len() as f64;
    let source_environment_completion_overall_rsr = scenario_scores
        .iter()
        .map(|score| score.source_environment_completion_rsr)
        .sum::<f64>()
        / scenario_count;
    let policy_recovery_overall_rsr = scenario_scores
        .iter()
        .map(|score| score.policy_recovery_rsr)
        .sum::<f64>()
        / scenario_count;
    let physical_safe_overall_rsr = scenario_scores
        .iter()
        .map(|score| score.physical_safe_rsr)
        .sum::<f64>()
        / scenario_count;

    // Each declared occupied Stage–Type group receives equal weight. Within a
    // group, scenarios receive equal weight regardless of rollout count.
    let mut group_scores = Vec::with_capacity(manifest.macro9_groups.len());
    for group in &manifest.macro9_groups {
        let members = scenario_scores
            .iter()
            .filter(|score| score.stage == group.stage && score.deviation == group.deviation)
            .collect::<Vec<_>>();
        let member_count = members.len() as f64;
        let group_source_environment = members
            .iter()
            .map(|score| score.source_environment_completion_rsr)
            .sum::<f64>()
            / member_count;
        let group_recovery = members
            .iter()
            .map(|score| score.policy_recovery_rsr)
            .sum::<f64>()
            / member_count;
        let group_safe = members
            .iter()
            .map(|score| score.physical_safe_rsr)
            .sum::<f64>()
            / member_count;

        group_scores.push(GroupScore {
            group: *group,
            scenario_count: members.len(),
            source_environment_completion_rsr: group_source_environment,
            policy_recovery_rsr: group_recovery,
            physical_safe_rsr: group_safe,
        });
    }

    let (
        source_environment_completion_macro9_rsr,
        policy_recovery_macro9_rsr,
        physical_safe_macro9_rsr,
    ) = if manifest.split == BenchmarkSplit::Test {
        (
            Some(
                group_scores.iter().map(|score| score.source_environment_completion_rsr).sum::<f64>()
                    / 9.0,
            ),
            Some(
                group_scores.iter().map(|score| score.policy_recovery_rsr).sum::<f64>() / 9.0,
            ),
            Some(
                group_scores.iter().map(|score| score.physical_safe_rsr).sum::<f64>() / 9.0,
            ),
        )
    } else {
        (None, None, None)
    };

    let accepted_ids = trials
        .iter()
        .filter(|trial| {
            !duplicate_ids.contains(&trial.evidence.trial_id)
                && expected_to_scenario
                    .get(&trial.evidence.trial_id)
                    .is_some_and(|scenario| {
                        trial.platform == manifest.platform
                            && trial.split == manifest.split
                            && trial.execution_mode == manifest.execution_mode
                            && trial.source_repository == manifest.source_repository
                            && trial.source_revision == manifest.source_revision
                            && trial.split_manifest_blob_sha == manifest.split_manifest_blob_sha
                            && trial.evidence.policy_digest == manifest.policy_digest
                            && trial.evidence.arm_id == manifest.arm_id
                            && trial.stage == scenario.stage
                            && trial.deviation == scenario.deviation
                            && trial.source == scenario.source
                            && trial.evidence.scenario_id == scenario.scenario_id
                            && trial.source_path == scenario.source_path
                            && trial.scenario_content_sha256 == scenario.scenario_content_sha256
                            && trial.original_trajectory_sha256 == scenario.original_trajectory_sha256
                            && trial.task_name == scenario.task_name
                            && trial.actions == scenario.actions
                            && trial.replay_prefix_steps == scenario.replay_prefix_steps
                            && trial.evidence.scenario_digest == scenario.scenario_content_sha256
                            && trial.evidence.evaluator_digest == manifest.evaluator_digest
                            && trial.evidence.has_complete_identity()
                    })
        })
        .map(|trial| trial.evidence.trial_id.clone())
        .collect::<BTreeSet<_>>();
    let missing_expected_trial_ids = expected_to_scenario
        .keys()
        .filter(|trial_id| !accepted_ids.contains(*trial_id))
        .cloned()
        .collect();

    Ok(RoboRecoverScoreReport {
        platform: manifest.platform,
        split: manifest.split,
        execution_mode: manifest.execution_mode,
        source_repository: manifest.source_repository.clone(),
        source_revision: manifest.source_revision.clone(),
        split_manifest_blob_sha: manifest.split_manifest_blob_sha.clone(),
        policy_digest: manifest.policy_digest.clone(),
        evaluator_digest: manifest.evaluator_digest.clone(),
        arm_id: manifest.arm_id.clone(),
        scenario_count: manifest.scenarios.len(),
        expected_trial_count: expected_trial_ids.len(),
        observed_trial_rows: trials.len(),
        accepted_trial_rows,
        metadata_mismatch_rows,
        source_environment_completions,
        replay_preempted_rows,
        policy_recovery_successes,
        policy_not_completed_rows,
        unknown_policy_outcome_rows,
        contradictory_policy_outcome_rows,
        physical_safe_successes,
        missing_expected_trial_ids,
        duplicate_trial_ids: audit.duplicate_trial_ids.clone(),
        identity_audit: audit,
        scenarios: scenario_scores,
        groups: group_scores,
        source_environment_completion_overall_rsr,
        source_environment_completion_macro9_rsr,
        policy_recovery_overall_rsr,
        policy_recovery_macro9_rsr,
        physical_safe_overall_rsr,
        physical_safe_macro9_rsr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::SafetyEvidence;

    fn all_groups() -> Vec<StageTypeGroup> {
        vec![
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::WrongTarget },
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::StageRollback },
            StageTypeGroup { stage: RecoveryStage::Approach, deviation: DeviationType::SceneChange },
            StageTypeGroup { stage: RecoveryStage::Contact, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Move, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::WrongTarget },
            StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Place, deviation: DeviationType::SceneChange },
        ]
    }

    fn manifest() -> RoboRecoverManifest {
        let groups = all_groups();
        let scenarios = groups
            .iter()
            .enumerate()
            .map(|(index, group)| PlannedScenario {
                scenario_id: format!("scenario-{index}"),
                source_path: format!("robotwin/test/scenario-{index}.json"),
                scenario_content_sha256: format!("{index:064x}"),
                original_trajectory_sha256: format!("{:064x}", index + 1000),
                task_name: format!("task-{index}"),
                actions: 12,
                replay_prefix_steps: 12,
                source: ScenarioSource::Natural,
                stage: group.stage,
                deviation: group.deviation,
                expected_trial_ids: vec![format!("trial-{index}-0")],
            })
            .collect();
        RoboRecoverManifest {
            platform: BenchmarkPlatform::RoboTwin,
            split: BenchmarkSplit::Test,
            execution_mode: ExecutionMode::ReplayThenInfer,
            source_repository: "RUCKBReasoning/RoboRecover".to_owned(),
            source_revision: "089b9420e8b7c303e2338a8212a8b6c27c26a79b".to_owned(),
            split_manifest_blob_sha: "c9b40e12d6f8d2dab723815dfc2e966ea7e62cd5".to_owned(),
            policy_digest: "d".repeat(64),
            evaluator_digest: "e335523db8ad4a2e06def1180f0e734334dbbd3b873df69d46873e17face700d".to_owned(),
            arm_id: "base-policy".to_owned(),
            scenarios,
            macro9_groups: groups,
        }
    }

    fn sample(
        scenario_index: usize,
        trial_index: usize,
        terminal: TerminalOutcome,
        safety: SafetyEvidence,
        intervention: super::super::InterventionKind,
        world_state: super::super::WorldStateContinuity,
    ) -> RoboRecoverTrial {
        let manifest = manifest();
        let scenario = &manifest.scenarios[scenario_index];
        RoboRecoverTrial {
            platform: manifest.platform,
            split: manifest.split,
            execution_mode: manifest.execution_mode,
            source_repository: manifest.source_repository.clone(),
            source_path: scenario.source_path.clone(),
            source_revision: manifest.source_revision.clone(),
            split_manifest_blob_sha: manifest.split_manifest_blob_sha.clone(),
            stage: scenario.stage,
            deviation: scenario.deviation,
            scenario_content_sha256: scenario.scenario_content_sha256.clone(),
            original_trajectory_sha256: scenario.original_trajectory_sha256.clone(),
            task_name: scenario.task_name.clone(),
            actions: scenario.actions,
            replay_prefix_steps: scenario.replay_prefix_steps,
            source: scenario.source,
            source_status: "infer_ok".to_owned(),
            inference_steps: 3,
            source_terminal_success: Some(true),
            evidence: TrialEvidence {
                trial_id: format!("trial-{scenario_index}-{trial_index}"),
                scenario_id: scenario.scenario_id.clone(),
                arm_id: "base-policy".to_owned(),
                scenario_digest: format!("{scenario_index:064x}"),
                evaluator_digest: manifest().evaluator_digest,
                policy_digest: manifest().policy_digest,
                intervention,
                world_state,
                executed_action_count: 3,
                terminal,
                safety,
            },
        }
    }

    fn failed_trial(scenario_index: usize, trial_index: usize) -> RoboRecoverTrial {
        let mut trial = sample(
            scenario_index,
            trial_index,
            TerminalOutcome::TaskFailed,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.source_status = "infer_failed".to_owned();
        trial.source_terminal_success = Some(false);
        trial
    }

    #[test]
    fn policy_recovery_rsr_is_scenario_equal_not_episode_weighted() {
        let mut m = manifest();
        m.scenarios[0].expected_trial_ids = vec!["a".into()];
        m.scenarios[1].expected_trial_ids = vec!["b0".into(), "b1".into(), "b2".into()];

        let mut trials = vec![sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        )];
        trials[0].evidence.trial_id = "a".into();

        for (index, terminal) in [
            TerminalOutcome::TaskCompleted,
            TerminalOutcome::TaskFailed,
            TerminalOutcome::TaskFailed,
        ]
        .into_iter()
        .enumerate()
        {
            let mut trial = failed_trial(1, index);
            trial.evidence.trial_id = format!("b{index}");
            trial.evidence.terminal = terminal;
            if terminal == TerminalOutcome::TaskCompleted {
                trial.source_status = "infer_ok".to_owned();
                trial.inference_steps = 2;
                trial.source_terminal_success = Some(true);
            }
            trials.push(trial);
        }

        let report = score_roborecover(&m, &trials).unwrap();
        // (1.0 + 1/3 + seven zero-rate scenarios) / 9 = 0.148148...
        assert!((report.policy_recovery_overall_rsr - (4.0 / 27.0)).abs() < 1e-12);
        // Episode-weighted success would be 2/4 = 0.5 and is not this metric.
        assert!((report.policy_recovery_overall_rsr - 0.5).abs() > 0.1);
    }

    #[test]
    fn policy_recovery_success_does_not_imply_physical_safe_success() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::ViolationObserved,
            super::super::InterventionKind::CompensatingAction,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.policy_recovery_successes, 1);
        assert_eq!(report.physical_safe_successes, 0);
        assert!((report.policy_recovery_overall_rsr - (1.0 / 9.0)).abs() < 1e-12);
        assert_eq!(report.physical_safe_overall_rsr, 0.0);
        assert_eq!(report.identity_audit.safety_violation_rows, 1);
    }

    #[test]
    fn simulator_rewind_policy_success_is_not_physical_safe_success() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::SimulatorStateRewind,
            super::super::WorldStateContinuity::SimulatorRewound,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.policy_recovery_successes, 1);
        assert_eq!(report.physical_safe_successes, 0);
        assert_eq!(report.identity_audit.simulator_rewind_rows, 1);
    }

    #[test]
    fn missing_trials_remain_in_expected_denominator() {
        let m = manifest();
        let report = score_roborecover(&m, &[]).unwrap();
        assert_eq!(report.expected_trial_count, 9);
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.missing_expected_trial_ids.len(), 9);
        assert_eq!(report.policy_recovery_overall_rsr, 0.0);
        assert_eq!(report.physical_safe_overall_rsr, 0.0);
    }

    #[test]
    fn duplicate_observed_trial_ids_are_all_excluded() {
        let m = manifest();
        let mut first = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        first.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        let mut second = first.clone();
        second.evidence.arm_id = "sidecar".to_owned();

        let report = score_roborecover(&m, &[first, second]).unwrap();
        assert_eq!(report.duplicate_trial_ids.len(), 1);
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.missing_expected_trial_ids.len(), 9);
    }

    #[test]
    fn stage_metadata_mismatch_is_not_accepted_as_policy_success() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.stage = RecoveryStage::Place;

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.metadata_mismatch_rows, 1);
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.missing_expected_trial_ids.len(), 9);
    }

    #[test]
    fn platform_or_split_mismatch_is_not_accepted() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.split = BenchmarkSplit::Train;

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.metadata_mismatch_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
    }

    #[test]
    fn duplicate_or_missing_macro9_groups_invalidate_manifest() {
        let mut m = manifest();
        m.macro9_groups.pop();
        let errors = validate_manifest(&m).unwrap_err();
        assert!(errors.contains(&ManifestIssue::Macro9GroupCount(8)));

        let mut m = manifest();
        m.macro9_groups[0] = m.macro9_groups[1];
        let errors = validate_manifest(&m).unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            ManifestIssue::DuplicateMacro9Group(_)
        )));
    }

    #[test]
    fn train_and_test_splits_cannot_be_mixed() {
        let mut m = manifest();
        m.split = BenchmarkSplit::Train;
        m.macro9_groups.clear();
        let mut test_trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        test_trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        let report = score_roborecover(&m, &[test_trial]).unwrap();
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
    }
    #[test]
    fn different_policy_or_intervention_arm_cannot_be_pooled() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.evidence.arm_id = "monitor-correction".to_owned();

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
    }

    #[test]
    fn all_planned_scenarios_must_belong_to_declared_macro9_groups() {
        let mut m = manifest();
        m.scenarios[0].stage = RecoveryStage::Move;
        m.scenarios[0].deviation = DeviationType::WrongTarget;
        let errors = validate_manifest(&m).unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            ManifestIssue::ScenarioGroupNotInMacro9(_)
        )));
    }


    #[test]
    fn robo_twin_replay_preemption_is_not_policy_recovery() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.source_status = "replay_already_done".to_owned();
        trial.inference_steps = 0;
        trial.source_terminal_success = Some(true);

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.source_environment_completions, 1);
        assert!((report.source_environment_completion_overall_rsr - (1.0 / 9.0)).abs() < 1e-12);
        assert_eq!(report.source_environment_completion_macro9_rsr, Some(1.0 / 9.0));
        assert_eq!(report.replay_preempted_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.source_environment_completion_macro9_rsr, Some(1.0 / 9.0));
        assert_eq!(report.policy_recovery_macro9_rsr, Some(0.0));
        assert_eq!(report.physical_safe_successes, 0);
        assert_eq!(report.scenarios[0].policy_recovery_rsr, 0.0);
    }

    #[test]
    fn robo_twin_policy_success_requires_infer_ok_steps_and_terminal_flag() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::CompensatingAction,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.source_status = "infer_ok".to_owned();
        trial.inference_steps = 2;
        trial.source_terminal_success = Some(true);

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.policy_recovery_successes, 1);
        assert_eq!(report.physical_safe_successes, 1);
    }

    #[test]
    fn contradictory_robo_twin_success_fields_are_not_recovery_success() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.source_status = "infer_ok".to_owned();
        trial.inference_steps = 2;
        trial.source_terminal_success = Some(false);

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.contradictory_policy_outcome_rows, 1);
        assert_eq!(report.physical_safe_successes, 0);
    }

    #[test]
    fn libero_success_status_requires_inference_and_replay_completion_is_separate() {
        let mut m = manifest();
        m.platform = BenchmarkPlatform::Libero;
        m.split_manifest_blob_sha = "925af3d5ab5e810cd65b86f6d4f35bbc3891d67f".to_owned();
        m.evaluator_digest = "3e1cd54025389f4fb2b3e60c9bceb7e3ecafaddfc0e8ac4092f1e0313d14c9b1".to_owned();
        let mut success = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        success.platform = BenchmarkPlatform::Libero;
        success.split_manifest_blob_sha = m.split_manifest_blob_sha.clone();
        success.evidence.evaluator_digest = m.evaluator_digest.clone();
        success.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        success.source_status = "success".to_owned();
        success.inference_steps = 1;
        success.source_terminal_success = None;

        let mut replay = sample(
            1,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        replay.platform = BenchmarkPlatform::Libero;
        replay.split_manifest_blob_sha = m.split_manifest_blob_sha.clone();
        replay.evidence.evaluator_digest = m.evaluator_digest.clone();
        replay.evidence.trial_id = m.scenarios[1].expected_trial_ids[0].clone();
        replay.source_status = "replay_already_done".to_owned();
        replay.inference_steps = 0;
        replay.source_terminal_success = None;

        let report = score_roborecover(&m, &[success, replay]).unwrap();
        assert_eq!(report.policy_recovery_successes, 1);
        assert_eq!(report.replay_preempted_rows, 1);
        assert_eq!(report.source_environment_completions, 2);
        assert!((report.source_environment_completion_overall_rsr - (2.0 / 9.0)).abs() < 1e-12);
        assert!((report.policy_recovery_overall_rsr - (1.0 / 9.0)).abs() < 1e-12);
    }

    #[test]
    fn unknown_source_status_remains_accepted_but_is_not_success() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::Unknown,
            super::super::InterventionKind::Unknown,
            super::super::WorldStateContinuity::Unknown,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.source_status = "new_unrecognized_status".to_owned();
        trial.inference_steps = 1;
        trial.source_terminal_success = None;

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.accepted_trial_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.policy_not_completed_rows, 0);
        assert_eq!(report.unknown_policy_outcome_rows, 1);
        assert_eq!(report.physical_safe_successes, 0);
    }


    #[test]
    fn robo_twin_infer_ok_without_inference_actions_is_contradictory() {
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.source_status = "infer_ok".to_owned();
        trial.inference_steps = 0;
        trial.source_terminal_success = Some(true);

        assert_eq!(
            classify_policy_completion(&trial),
            PolicyCompletion::Contradictory
        );
        assert_eq!(source_reports_environment_completion(&trial), Some(true));
    }

    #[test]
    fn policy_success_status_with_non_success_terminal_record_is_contradictory() {
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskFailed,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.source_status = "infer_ok".to_owned();
        trial.inference_steps = 2;
        trial.source_terminal_success = Some(true);

        assert_eq!(
            classify_policy_completion(&trial),
            PolicyCompletion::Contradictory
        );
    }


    #[test]
    fn source_and_scenario_provenance_require_exact_digest_formats() {
        let mut m = manifest();
        m.source_revision = "main".to_owned();
        assert!(validate_manifest(&m).unwrap_err().contains(&ManifestIssue::InvalidSourceRevision));

        let mut m = manifest();
        m.split_manifest_blob_sha = "latest".to_owned();
        assert!(validate_manifest(&m).unwrap_err().contains(&ManifestIssue::InvalidSplitManifestBlobSha));

        let mut m = manifest();
        m.policy_digest = "policy".to_owned();
        assert!(validate_manifest(&m).unwrap_err().contains(&ManifestIssue::InvalidPolicyDigest));

        let mut m = manifest();
        m.evaluator_digest = "sha256:evaluator".to_owned();
        assert!(validate_manifest(&m).unwrap_err().contains(&ManifestIssue::InvalidEvaluatorDigest));

        let mut m = manifest();
        m.scenarios[0].scenario_content_sha256 = "scenario-0".to_owned();
        assert!(validate_manifest(&m).unwrap_err().contains(
            &ManifestIssue::InvalidScenarioDigest("scenario-0".to_owned())
        ));
    }


    #[test]
    fn scenario_and_evaluator_digest_mismatches_block_confirmatory_scoring() {
        let m = manifest();
        let mut wrong_scenario = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        wrong_scenario.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        wrong_scenario.evidence.scenario_digest = "f".repeat(64);

        let report = score_roborecover(&m, &[wrong_scenario]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);

        let mut wrong_evaluator = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        wrong_evaluator.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        wrong_evaluator.evidence.evaluator_digest = "e".repeat(64);

        let report = score_roborecover(&m, &[wrong_evaluator]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
    }

    #[test]
    fn score_receipt_carries_source_split_evaluator_and_scenario_identity() {
        let m = manifest();
        let report = score_roborecover(&m, &[]).unwrap();
        assert_eq!(report.source_repository, m.source_repository);
        assert_eq!(report.source_revision, m.source_revision);
        assert_eq!(report.split_manifest_blob_sha, m.split_manifest_blob_sha);
        assert_eq!(report.policy_digest, m.policy_digest);
        assert_eq!(report.evaluator_digest, m.evaluator_digest);
        assert_eq!(report.scenarios[0].scenario_content_sha256, m.scenarios[0].scenario_content_sha256);
        assert_eq!(report.scenarios[0].source, m.scenarios[0].source);
    }


    #[test]
    fn training_split_allows_training_only_group_and_omits_macro9() {
        let mut m = manifest();
        m.split = BenchmarkSplit::Train;
        m.split_manifest_blob_sha = "b211728e96d4e6548d4d5353daf73346c37f5a4d".to_owned();
        m.macro9_groups.clear();
        m.scenarios[0].stage = RecoveryStage::Move;
        m.scenarios[0].deviation = DeviationType::WrongTarget;

        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.split = BenchmarkSplit::Train;
        trial.split_manifest_blob_sha = m.split_manifest_blob_sha.clone();
        trial.stage = RecoveryStage::Move;
        trial.deviation = DeviationType::WrongTarget;
        trial.evidence.scenario_id = m.scenarios[0].scenario_id.clone();
        trial.evidence.scenario_digest = m.scenarios[0].scenario_content_sha256.clone();
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.policy_recovery_successes, 1);
        assert_eq!(report.policy_recovery_macro9_rsr, None);
        assert_eq!(report.source_environment_completion_macro9_rsr, None);
        assert_eq!(report.physical_safe_macro9_rsr, None);
        assert!(report.groups.is_empty());
    }

    #[test]
    fn training_split_rejects_macro9_declarations() {
        let mut m = manifest();
        m.split = BenchmarkSplit::Train;
        assert!(validate_manifest(&m)
            .unwrap_err()
            .contains(&ManifestIssue::Macro9OnlyAppliesToTest));
    }

    #[test]
    fn source_revision_and_split_manifest_mismatch_block_trial_admission() {
        let m = manifest();
        let mut wrong_revision = sample(
            0, 0, TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        wrong_revision.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        wrong_revision.source_revision = "a".repeat(40);

        let report = score_roborecover(&m, &[wrong_revision]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);

        let mut wrong_split_manifest = sample(
            0, 0, TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        wrong_split_manifest.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        wrong_split_manifest.split_manifest_blob_sha = "f".repeat(40);

        let report = score_roborecover(&m, &[wrong_split_manifest]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
    }

    #[test]
    fn different_policy_checkpoint_cannot_be_pooled_under_same_arm_label() {
        let m = manifest();
        let mut trial = sample(
            0, 0, TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.evidence.policy_digest = "f".repeat(64);

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
    }


    #[test]
    fn clean_seed_only_runs_are_not_recovery_evidence() {
        let mut m = manifest();
        m.execution_mode = ExecutionMode::SeedOnly;
        assert!(validate_manifest(&m)
            .unwrap_err()
            .contains(&ManifestIssue::SeedOnlyIsNotRecovery));
    }

    #[test]
    fn execution_mode_mismatch_blocks_trial_admission() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.execution_mode = ExecutionMode::SeedOnly;
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
    }


    #[test]
    fn replay_already_done_with_inference_actions_is_contradictory() {
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.source_status = "replay_already_done".to_owned();
        trial.inference_steps = 1;
        trial.source_terminal_success = Some(true);

        assert_eq!(
            classify_policy_completion(&trial),
            PolicyCompletion::Contradictory
        );
    }

    #[test]
    fn replay_only_status_is_not_credited_as_policy_success() {
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.source_status = "replay_ok".to_owned();
        trial.inference_steps = 0;
        trial.source_terminal_success = Some(true);

        assert_eq!(
            classify_policy_completion(&trial),
            PolicyCompletion::ReplayPreempted
        );
    }


    #[test]
    fn known_policy_failure_is_reported_separately_from_unknown_or_contradictory() {
        let m = manifest();
        let mut trial = failed_trial(
            0,
            0,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.accepted_trial_rows, 1);
        assert_eq!(report.policy_recovery_successes, 0);
        assert_eq!(report.policy_not_completed_rows, 1);
        assert_eq!(report.unknown_policy_outcome_rows, 0);
        assert_eq!(report.contradictory_policy_outcome_rows, 0);
    }


    #[test]
    fn source_repository_and_construction_source_must_match_manifest() {
        let m = manifest();
        let mut wrong_repository = sample(
            0, 0, TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        wrong_repository.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        wrong_repository.source_repository = "another/benchmark".to_owned();

        let report = score_roborecover(&m, &[wrong_repository]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);

        let mut wrong_source = sample(
            0, 0, TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        wrong_source.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        wrong_source.source = ScenarioSource::HumanConstructed;

        let report = score_roborecover(&m, &[wrong_source]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
    }


    #[test]
    fn unknown_zero_action_status_is_not_assumed_to_be_replay_preemption() {
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::Unknown,
            super::super::InterventionKind::Unknown,
            super::super::WorldStateContinuity::Unknown,
        );
        trial.source_status = "future_status".to_owned();
        trial.inference_steps = 0;
        trial.source_terminal_success = Some(true);

        assert_eq!(
            classify_policy_completion(&trial),
            PolicyCompletion::Unknown
        );
    }


    #[test]
    fn source_construction_classification_must_match_frozen_scenario_record() {
        let m = manifest();
        let mut trial = sample(
            0,
            0,
            TerminalOutcome::TaskCompleted,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        );
        trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
        trial.source = ScenarioSource::HumanConstructed;

        let report = score_roborecover(&m, &[trial]).unwrap();
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
    }

    #[test]
    fn source_path_original_trajectory_task_and_replay_metadata_are_bound() {
        let m = manifest();
        for field in 0..4 {
            let mut trial = sample(
                0, 0, TerminalOutcome::TaskCompleted,
                SafetyEvidence::VerifiedCompliant,
                super::super::InterventionKind::DirectPolicy,
                super::super::WorldStateContinuity::Preserved,
            );
            trial.evidence.trial_id = m.scenarios[0].expected_trial_ids[0].clone();
            match field {
                0 => trial.source_path.push_str(".tampered"),
                1 => trial.original_trajectory_sha256 = "f".repeat(64),
                2 => trial.task_name.push_str("-tampered"),
                3 => trial.replay_prefix_steps += 1,
                _ => unreachable!(),
            }

            let report = score_roborecover(&m, &[trial]).unwrap();
            assert_eq!(report.accepted_trial_rows, 0);
            assert_eq!(report.metadata_mismatch_rows, 1);
            assert_eq!(report.policy_recovery_successes, 0);
        }
    }

}
