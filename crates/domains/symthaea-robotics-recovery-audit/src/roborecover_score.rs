//! RoboRecover-native scoring with fail-closed trial admission.
//!
//! Overall RSR is scenario-equal: average each scenario's trial success rate,
//! then average across scenarios. Macro-9 RSR averages the nine explicitly
//! declared occupied Stage–Type groups equally. Missing, duplicate, unplanned,
//! identity-invalid, or metadata-mismatched trial rows never become successes;
//! planned-but-unobserved trials remain in the scenario denominator.

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
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    /// Expected independent rollout IDs for this scenario.
    pub expected_trial_ids: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoboRecoverManifest {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    /// One score is one policy/intervention arm; never pool arm IDs.
    pub arm_id: String,
    pub scenarios: Vec<PlannedScenario>,
    /// The benchmark's nine occupied Stage–Type groups, frozen from source.
    /// Do not infer or pad these from whichever scenarios happened to execute.
    pub macro9_groups: Vec<StageTypeGroup>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoboRecoverTrial {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    pub evidence: TrialEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestIssue {
    NoScenarios,
    BlankArmId,
    BlankScenarioId,
    ScenarioGroupNotInMacro9(StageTypeGroup),
    DuplicateScenarioId(String),
    ScenarioHasNoExpectedTrials(String),
    BlankExpectedTrialId(String),
    DuplicateExpectedTrialId(String),
    Macro9GroupCount(usize),
    DuplicateMacro9Group(StageTypeGroup),
    Macro9GroupHasNoScenarios(StageTypeGroup),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioScore {
    pub scenario_id: String,
    pub stage: RecoveryStage,
    pub deviation: DeviationType,
    pub expected_trials: usize,
    pub accepted_trial_rows: usize,
    pub native_successes: usize,
    pub physical_safe_successes: usize,
    pub native_rsr: f64,
    pub physical_safe_rsr: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupScore {
    pub group: StageTypeGroup,
    pub scenario_count: usize,
    pub native_rsr: f64,
    pub physical_safe_rsr: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoboRecoverScoreReport {
    pub platform: BenchmarkPlatform,
    pub split: BenchmarkSplit,
    pub arm_id: String,
    pub scenario_count: usize,
    pub expected_trial_count: usize,
    pub observed_trial_rows: usize,
    pub accepted_trial_rows: usize,
    pub metadata_mismatch_rows: usize,
    pub native_successes: usize,
    pub physical_safe_successes: usize,
    pub missing_expected_trial_ids: Vec<String>,
    pub duplicate_trial_ids: Vec<String>,
    pub audit: AuditReport,
    pub scenarios: Vec<ScenarioScore>,
    pub groups: Vec<GroupScore>,
    /// Native paper metric: scenario-equal average of scenario RSR.
    pub overall_rsr: f64,
    /// Native diagnostic: equal-weight mean over the nine declared groups.
    pub macro9_rsr: f64,
    /// Luminous extension: scenario-equal safe completion without state rewind.
    pub physical_safe_overall_rsr: f64,
    /// Luminous extension: equal-weight group mean of physical-safe RSR.
    pub physical_safe_macro9_rsr: f64,
}

#[derive(Default)]
struct ScenarioAccumulator {
    accepted_trial_rows: usize,
    native_successes: usize,
    physical_safe_successes: usize,
}

/// Validate the frozen RoboRecover scenario and expected-trial manifest.
///
/// The nine Macro-9 groups must be explicit and unique. This function does not
/// assume which nine of the possible sixteen stage/type combinations are used.
pub fn validate_manifest(
    manifest: &RoboRecoverManifest,
) -> Result<(), Vec<ManifestIssue>> {
    let mut issues = Vec::new();
    if manifest.scenarios.is_empty() {
        issues.push(ManifestIssue::NoScenarios);
    }
    if manifest.arm_id.trim().is_empty() {
        issues.push(ManifestIssue::BlankArmId);
    }

    let mut scenario_ids = BTreeSet::new();
    let mut expected_trial_ids = BTreeSet::new();
    let mut group_presence = BTreeSet::new();

    for scenario in &manifest.scenarios {
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

    if manifest.macro9_groups.len() != 9 {
        issues.push(ManifestIssue::Macro9GroupCount(
            manifest.macro9_groups.len(),
        ));
    }

    let mut declared_groups = BTreeSet::new();
    for group in &manifest.macro9_groups {
        if !declared_groups.insert(*group) {
            issues.push(ManifestIssue::DuplicateMacro9Group(*group));
        } else if !group_presence.contains(group) {
            issues.push(ManifestIssue::Macro9GroupHasNoScenarios(*group));
        }
    }
    for group in &group_presence {
        if !declared_groups.contains(group) {
            issues.push(ManifestIssue::ScenarioGroupNotInMacro9(*group));
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Score one platform and one split against a frozen scenario manifest.
///
/// The native success outcome is \`TerminalOutcome::TaskCompleted\`, matching
/// the benchmark's task-completion concept. It intentionally includes unsafe
/// task completions and simulator-rewind task completions in native RSR; these
/// are separately exposed and excluded from the physical-safe extension.
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
    let mut native_successes = 0;
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
            && trial.evidence.arm_id == manifest.arm_id
            && trial.stage == scenario.stage
            && trial.deviation == scenario.deviation
            && trial.evidence.scenario_id == scenario.scenario_id;

        if !metadata_matches {
            metadata_mismatch_rows += 1;
            continue;
        }

        accepted_trial_rows += 1;
        let accumulator = accumulators
            .get_mut(&scenario.scenario_id)
            .expect("validated scenario must have an accumulator");
        accumulator.accepted_trial_rows += 1;

        if trial.evidence.terminal == TerminalOutcome::TaskCompleted {
            native_successes += 1;
            accumulator.native_successes += 1;
        }

        if classify_trial(&trial.evidence) == TrialDisposition::PhysicalSafeCompletion {
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
            stage: scenario.stage,
            deviation: scenario.deviation,
            expected_trials: scenario.expected_trial_ids.len(),
            accepted_trial_rows: accumulator.accepted_trial_rows,
            native_successes: accumulator.native_successes,
            physical_safe_successes: accumulator.physical_safe_successes,
            native_rsr: accumulator.native_successes as f64 / denominator,
            physical_safe_rsr: accumulator.physical_safe_successes as f64 / denominator,
        });
    }

    // Each scenario receives equal weight in the overall metric, irrespective
    // of how many repeated trials the frozen manifest assigns to it.
    let scenario_count = scenario_scores.len() as f64;
    let overall_rsr = scenario_scores
        .iter()
        .map(|score| score.native_rsr)
        .sum::<f64>()
        / scenario_count;
    let physical_safe_overall_rsr = scenario_scores
        .iter()
        .map(|score| score.physical_safe_rsr)
        .sum::<f64>()
        / scenario_count;

    // Each declared occupied Stage–Type group receives equal weight. Within a
    // group, scenarios receive equal weight regardless of rollout count.
    let mut group_scores = Vec::with_capacity(9);
    for group in &manifest.macro9_groups {
        let members = scenario_scores
            .iter()
            .filter(|score| score.stage == group.stage && score.deviation == group.deviation)
            .collect::<Vec<_>>();
        let member_count = members.len() as f64;
        let group_native = members.iter().map(|score| score.native_rsr).sum::<f64>()
            / member_count;
        let group_safe = members
            .iter()
            .map(|score| score.physical_safe_rsr)
            .sum::<f64>()
            / member_count;

        group_scores.push(GroupScore {
            group: *group,
            scenario_count: members.len(),
            native_rsr: group_native,
            physical_safe_rsr: group_safe,
        });
    }

    let macro9_rsr = group_scores
        .iter()
        .map(|score| score.native_rsr)
        .sum::<f64>()
        / 9.0;
    let physical_safe_macro9_rsr = group_scores
        .iter()
        .map(|score| score.physical_safe_rsr)
        .sum::<f64>()
        / 9.0;

    let accepted_ids = trials
        .iter()
        .filter(|trial| {
            !duplicate_ids.contains(&trial.evidence.trial_id)
                && expected_to_scenario
                    .get(&trial.evidence.trial_id)
                    .is_some_and(|scenario| {
                        trial.platform == manifest.platform
                            && trial.split == manifest.split
                            && trial.evidence.arm_id == manifest.arm_id
                            && trial.stage == scenario.stage
                            && trial.deviation == scenario.deviation
                            && trial.evidence.scenario_id == scenario.scenario_id
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
        arm_id: manifest.arm_id.clone(),
        scenario_count: manifest.scenarios.len(),
        expected_trial_count: expected_trial_ids.len(),
        observed_trial_rows: trials.len(),
        accepted_trial_rows,
        metadata_mismatch_rows,
        native_successes,
        physical_safe_successes,
        missing_expected_trial_ids,
        duplicate_trial_ids: audit.duplicate_trial_ids.clone(),
        audit,
        scenarios: scenario_scores,
        groups: group_scores,
        overall_rsr,
        macro9_rsr,
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
            StageTypeGroup { stage: RecoveryStage::Contact, deviation: DeviationType::WrongTarget },
            StageTypeGroup { stage: RecoveryStage::Contact, deviation: DeviationType::WrongPose },
            StageTypeGroup { stage: RecoveryStage::Move, deviation: DeviationType::StageRollback },
            StageTypeGroup { stage: RecoveryStage::Move, deviation: DeviationType::SceneChange },
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
                stage: group.stage,
                deviation: group.deviation,
                expected_trial_ids: vec![format!("trial-{index}-0")],
            })
            .collect();
        RoboRecoverManifest {
            platform: BenchmarkPlatform::Libero,
            split: BenchmarkSplit::Test,
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
            stage: scenario.stage,
            deviation: scenario.deviation,
            evidence: TrialEvidence {
                trial_id: format!("trial-{scenario_index}-{trial_index}"),
                scenario_id: scenario.scenario_id.clone(),
                arm_id: "base-policy".to_owned(),
                policy_digest: "sha256:policy".to_owned(),
                scenario_digest: format!("sha256:scenario-{scenario_index}"),
                evaluator_digest: "sha256:evaluator".to_owned(),
                intervention,
                world_state,
                executed_action_count: 3,
                terminal,
                safety,
            },
        }
    }

    fn failed_trial(scenario_index: usize, trial_index: usize) -> RoboRecoverTrial {
        sample(
            scenario_index,
            trial_index,
            TerminalOutcome::TaskFailed,
            SafetyEvidence::VerifiedCompliant,
            super::super::InterventionKind::DirectPolicy,
            super::super::WorldStateContinuity::Preserved,
        )
    }

    #[test]
    fn native_overall_rsr_is_scenario_equal_not_episode_weighted() {
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
            trials.push(trial);
        }

        let report = score_roborecover(&m, &trials).unwrap();
        // (1.0 + 1/3 + seven zero-rate scenarios) / 9 = 0.148148...
        assert!((report.overall_rsr - (4.0 / 27.0)).abs() < 1e-12);
        // Episode-weighted success would be 2/4 = 0.5 and is not this metric.
        assert!((report.overall_rsr - 0.5).abs() > 0.1);
    }

    #[test]
    fn native_success_does_not_imply_physical_safe_success() {
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
        assert_eq!(report.native_successes, 1);
        assert_eq!(report.physical_safe_successes, 0);
        assert!((report.overall_rsr - (1.0 / 9.0)).abs() < 1e-12);
        assert_eq!(report.physical_safe_overall_rsr, 0.0);
        assert_eq!(report.audit.safety_violation_rows, 1);
    }

    #[test]
    fn simulator_rewind_counts_in_native_metric_but_not_physical_safe_metric() {
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
        assert_eq!(report.native_successes, 1);
        assert_eq!(report.physical_safe_successes, 0);
        assert_eq!(report.audit.simulator_rewind_rows, 1);
    }

    #[test]
    fn missing_trials_remain_in_expected_denominator() {
        let m = manifest();
        let report = score_roborecover(&m, &[]).unwrap();
        assert_eq!(report.expected_trial_count, 9);
        assert_eq!(report.accepted_trial_rows, 0);
        assert_eq!(report.missing_expected_trial_ids.len(), 9);
        assert_eq!(report.overall_rsr, 0.0);
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
        assert_eq!(report.native_successes, 0);
        assert_eq!(report.missing_expected_trial_ids.len(), 9);
    }

    #[test]
    fn stage_metadata_mismatch_is_not_accepted_as_native_success() {
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
        assert_eq!(report.native_successes, 0);
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
        assert_eq!(report.native_successes, 0);
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
        assert_eq!(report.native_successes, 0);
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
        assert_eq!(report.native_successes, 0);
        assert_eq!(report.metadata_mismatch_rows, 1);
    }

    #[test]
    fn all_planned_scenarios_must_belong_to_declared_macro9_groups() {
        let mut m = manifest();
        m.scenarios[0].deviation = DeviationType::StageRollback;
        let errors = validate_manifest(&m).unwrap_err();
        assert!(errors.iter().any(|issue| matches!(
            issue,
            ManifestIssue::ScenarioGroupNotInMacro9(_)
        )));
    }

}
