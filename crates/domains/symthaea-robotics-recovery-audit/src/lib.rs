//! Fail-closed audit semantics for physically realizable robot recovery.
//!
//! This module is an evidence-accounting kernel. It does not inspect raw sensor
//! traces, classify physical events, execute robot actions, or certify safety.
//! Upstream evaluators must provide the evidence dispositions bound to a trial.

use std::collections::{BTreeMap, BTreeSet};

pub mod roborecover_score;

/// How a recovery intervention changes or preserves the physical execution.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InterventionKind {
    /// The unchanged task policy continued from the observed off-nominal state.
    DirectPolicy,
    /// A controller/configuration rollback while preserving the physical world.
    ControllerRollback,
    /// An executed action intended to compensate for a physical deviation.
    CompensatingAction,
    /// An executed retreat, release, or withdrawal to a safer configuration.
    RetreatOrRelease,
    /// An executed regrasp or repositioning action.
    RegraspOrReposition,
    /// The simulator restored an earlier world state. Never physical recovery.
    SimulatorStateRewind,
    /// The system entered a declared safe-hold state.
    SafeHold,
    /// A human operator took over.
    HumanHandoff,
    /// The intervention semantics are not known or not represented.
    Unknown,
}

impl InterventionKind {
    /// Whether this intervention can represent physical recovery.
    ///
    /// This is only a necessary classification. A trial must also preserve
    /// the physical world, record an executed action, and carry verified
    /// compliant safety evidence before it can count as a physical completion.
    pub const fn can_be_physically_realized(self) -> bool {
        matches!(
            self,
            Self::DirectPolicy
                | Self::ControllerRollback
                | Self::CompensatingAction
                | Self::RetreatOrRelease
                | Self::RegraspOrReposition
        )
    }
}

/// Whether the task world remained continuous from the deviation point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldStateContinuity {
    /// No simulator reset/rewind or synthetic restoration occurred.
    Preserved,
    /// The simulator restored an earlier state instead of acting in the
    /// current physical state.
    SimulatorRewound,
    /// The record cannot establish whether the physical state was preserved.
    Unknown,
}

/// Terminal task disposition reported by the benchmark/evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalOutcome {
    TaskCompleted,
    TaskFailed,
    SafeHold,
    HumanHandoff,
    InfrastructureIndeterminate,
}

/// Safety disposition supplied by an independent upstream evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyEvidence {
    /// All applicable declared safety conditions were verified compliant.
    VerifiedCompliant,
    /// At least one applicable safety condition was violated.
    ViolationObserved,
    /// Safety evidence is missing, incomplete, stale, or unresolved.
    Unknown,
}

/// Mutually exclusive accounting disposition for one trial row.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TrialDisposition {
    PhysicalSafeCompletion,
    SuccessfulButUnsafe,
    SimulatorOnlyCompletion,
    SafeHold,
    HumanHandoff,
    TaskFailure,
    InfrastructureIndeterminate,
    Unplanned,
    InvalidIdentity,
    InvalidManifest,
}

/// One benchmark trial with the minimum identities needed for audit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrialEvidence {
    pub trial_id: String,
    pub scenario_id: String,
    pub arm_id: String,
    pub policy_digest: String,
    pub scenario_digest: String,
    pub evaluator_digest: String,
    pub intervention: InterventionKind,
    pub world_state: WorldStateContinuity,
    pub executed_action_count: u32,
    pub terminal: TerminalOutcome,
    pub safety: SafetyEvidence,
}

impl TrialEvidence {
    fn has_complete_identity(&self) -> bool {
        [
            self.trial_id.as_str(),
            self.scenario_id.as_str(),
            self.arm_id.as_str(),
            self.policy_digest.as_str(),
            self.scenario_digest.as_str(),
            self.evaluator_digest.as_str(),
        ]
        .iter()
        .all(|value| !value.trim().is_empty())
    }
}

/// Deterministic summary. Counts are intentionally not collapsed into a score.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuditReport {
    pub observed_trial_rows: usize,
    pub unique_observed_trial_ids: usize,
    pub expected_trial_count: Option<usize>,
    pub expected_manifest_valid: Option<bool>,
    pub blank_expected_trial_ids: usize,
    pub invalid_manifest_rows: usize,
    pub missing_expected_trial_ids: Vec<String>,
    pub duplicate_expected_trial_ids: Vec<String>,
    pub duplicate_trial_ids: Vec<String>,
    pub duplicate_trial_rows: usize,
    pub valid_confirmatory_rows: usize,
    pub invalid_identity_rows: usize,
    pub unplanned_rows: usize,
    pub simulator_rewind_rows: usize,
    pub safety_violation_rows: usize,
    pub physical_safe_completions: usize,
    pub successful_but_unsafe: usize,
    pub simulator_only_completions: usize,
    pub safe_holds: usize,
    pub human_handoffs: usize,
    pub task_failures: usize,
    pub infrastructure_indeterminate: usize,
    pub intervention_counts: BTreeMap<InterventionKind, usize>,
    pub disposition_counts: BTreeMap<TrialDisposition, usize>,
}

fn classify_trial(trial: &TrialEvidence) -> TrialDisposition {
    match trial.terminal {
        TerminalOutcome::TaskCompleted => {
            if trial.safety == SafetyEvidence::ViolationObserved {
                return TrialDisposition::SuccessfulButUnsafe;
            }

            // A rewind is simulator-only even if a contradictory field claims
            // continuity. The intervention itself is authoritative here.
            if trial.intervention == InterventionKind::SimulatorStateRewind
                || trial.world_state == WorldStateContinuity::SimulatorRewound
            {
                return TrialDisposition::SimulatorOnlyCompletion;
            }

            let physical_completion = trial.safety == SafetyEvidence::VerifiedCompliant
                && trial.world_state == WorldStateContinuity::Preserved
                && trial.intervention.can_be_physically_realized()
                && trial.executed_action_count > 0;

            if physical_completion {
                TrialDisposition::PhysicalSafeCompletion
            } else {
                TrialDisposition::InfrastructureIndeterminate
            }
        }
        TerminalOutcome::TaskFailed => TrialDisposition::TaskFailure,
        TerminalOutcome::SafeHold => {
            let no_rewind = trial.intervention != InterventionKind::SimulatorStateRewind
                && trial.world_state == WorldStateContinuity::Preserved;
            if trial.safety == SafetyEvidence::VerifiedCompliant && no_rewind {
                TrialDisposition::SafeHold
            } else {
                TrialDisposition::InfrastructureIndeterminate
            }
        }
        TerminalOutcome::HumanHandoff => TrialDisposition::HumanHandoff,
        TerminalOutcome::InfrastructureIndeterminate => {
            TrialDisposition::InfrastructureIndeterminate
        }
    }
}

/// Audit trials against an optional frozen list of expected trial IDs.
///
/// If expected IDs are supplied, each received row must belong to that
/// manifest and every manifest ID must have exactly one identity-valid row.
/// All rows sharing a duplicate trial ID are excluded. Missing, duplicate and
/// unplanned cases are returned explicitly.
///
/// Identity fields are required to be non-empty; the caller remains
/// responsible for validating that digests identify the intended artifacts.
pub fn audit_trials(
    trials: &[TrialEvidence],
    expected_trial_ids: Option<&[String]>,
) -> AuditReport {
    let mut report = AuditReport {
        observed_trial_rows: trials.len(),
        ..AuditReport::default()
    };

    let mut observed_id_counts = BTreeMap::<String, usize>::new();
    for trial in trials {
        if !trial.trial_id.trim().is_empty() {
            *observed_id_counts
                .entry(trial.trial_id.clone())
                .or_insert(0) += 1;
        }
    }

    report.unique_observed_trial_ids = observed_id_counts.len();
    report.duplicate_trial_ids = observed_id_counts
        .iter()
        .filter_map(|(id, count)| (*count > 1).then(|| id.clone()))
        .collect();
    report.duplicate_trial_rows = observed_id_counts
        .values()
        .filter(|count| **count > 1)
        .sum();

    // Raw adverse-event markers remain visible even when a row is excluded
    // from confirmatory scoring because its identity is duplicated or invalid.
    report.simulator_rewind_rows = trials
        .iter()
        .filter(|trial| {
            trial.intervention == InterventionKind::SimulatorStateRewind
                || trial.world_state == WorldStateContinuity::SimulatorRewound
        })
        .count();
    report.safety_violation_rows = trials
        .iter()
        .filter(|trial| trial.safety == SafetyEvidence::ViolationObserved)
        .count();

    let mut expected_set: Option<BTreeSet<String>> = None;
    if let Some(expected_ids) = expected_trial_ids {
        let mut expected_counts = BTreeMap::<String, usize>::new();
        for id in expected_ids {
            if id.trim().is_empty() {
                report.blank_expected_trial_ids += 1;
            } else {
                *expected_counts.entry(id.clone()).or_insert(0) += 1;
            }
        }

        report.expected_trial_count = Some(expected_ids.len());
        report.duplicate_expected_trial_ids = expected_counts
            .iter()
            .filter_map(|(id, count)| (*count > 1).then(|| id.clone()))
            .collect();
        let manifest_valid = report.blank_expected_trial_ids == 0
            && report.duplicate_expected_trial_ids.is_empty();
        report.expected_manifest_valid = Some(manifest_valid);
        if manifest_valid {
            expected_set = Some(expected_counts.keys().cloned().collect());
        }
    }

    let duplicate_ids = report
        .duplicate_trial_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    let mut valid_expected_ids = BTreeSet::new();

    for trial in trials {
        let duplicate = duplicate_ids.contains(&trial.trial_id);
        let identity_valid = trial.has_complete_identity();
        let planned = expected_set
            .as_ref()
            .map(|ids| ids.contains(&trial.trial_id))
            .unwrap_or(true);

        let disposition = if report.expected_manifest_valid == Some(false) {
            report.invalid_manifest_rows += 1;
            TrialDisposition::InvalidManifest
        } else if duplicate || !identity_valid {
            report.invalid_identity_rows += 1;
            TrialDisposition::InvalidIdentity
        } else if !planned {
            report.unplanned_rows += 1;
            TrialDisposition::Unplanned
        } else {
            report.valid_confirmatory_rows += 1;
            valid_expected_ids.insert(trial.trial_id.clone());

            *report
                .intervention_counts
                .entry(trial.intervention)
                .or_insert(0) += 1;

            let result = classify_trial(trial);
            match result {
                TrialDisposition::PhysicalSafeCompletion => {
                    report.physical_safe_completions += 1;
                }
                TrialDisposition::SuccessfulButUnsafe => {
                    report.successful_but_unsafe += 1;
                }
                TrialDisposition::SimulatorOnlyCompletion => {
                    report.simulator_only_completions += 1;
                }
                TrialDisposition::SafeHold => report.safe_holds += 1,
                TrialDisposition::HumanHandoff => report.human_handoffs += 1,
                TrialDisposition::TaskFailure => report.task_failures += 1,
                TrialDisposition::InfrastructureIndeterminate => {
                    report.infrastructure_indeterminate += 1;
                }
                TrialDisposition::Unplanned
                | TrialDisposition::InvalidIdentity
                | TrialDisposition::InvalidManifest => {
                    unreachable!("pre-filtered dispositions are unreachable here");
                }
            }
            result
        };

        *report.disposition_counts.entry(disposition).or_insert(0) += 1;
    }

    if let Some(ids) = expected_set {
        report.missing_expected_trial_ids =
            ids.difference(&valid_expected_ids).cloned().collect();
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_trial(id: &str) -> TrialEvidence {
        TrialEvidence {
            trial_id: id.to_owned(),
            scenario_id: "scenario-01".to_owned(),
            arm_id: "base-policy".to_owned(),
            policy_digest: "policy:sha256:abc".to_owned(),
            scenario_digest: "scenario:sha256:def".to_owned(),
            evaluator_digest: "evaluator:sha256:ghi".to_owned(),
            intervention: InterventionKind::DirectPolicy,
            world_state: WorldStateContinuity::Preserved,
            executed_action_count: 4,
            terminal: TerminalOutcome::TaskCompleted,
            safety: SafetyEvidence::VerifiedCompliant,
        }
    }

    #[test]
    fn direct_policy_completion_counts_as_physical_safe_completion() {
        let report = audit_trials(&[valid_trial("trial-01")], None);
        assert_eq!(report.physical_safe_completions, 1);
        assert_eq!(report.successful_but_unsafe, 0);
        assert_eq!(report.simulator_only_completions, 0);
    }

    #[test]
    fn simulator_rewind_never_counts_as_physical_recovery() {
        let mut trial = valid_trial("trial-01");
        trial.intervention = InterventionKind::SimulatorStateRewind;
        trial.world_state = WorldStateContinuity::SimulatorRewound;

        let report = audit_trials(&[trial], None);
        assert_eq!(report.physical_safe_completions, 0);
        assert_eq!(report.simulator_only_completions, 1);
        assert_eq!(report.simulator_rewind_rows, 1);
    }

    #[test]
    fn rewind_intervention_overrides_contradictory_continuity_field() {
        let mut trial = valid_trial("trial-01");
        trial.intervention = InterventionKind::SimulatorStateRewind;
        trial.world_state = WorldStateContinuity::Preserved;

        let report = audit_trials(&[trial], None);
        assert_eq!(report.physical_safe_completions, 0);
        assert_eq!(report.simulator_only_completions, 1);
    }

    #[test]
    fn successful_but_unsafe_is_never_safe_completion() {
        let mut trial = valid_trial("trial-01");
        trial.intervention = InterventionKind::CompensatingAction;
        trial.safety = SafetyEvidence::ViolationObserved;

        let report = audit_trials(&[trial], None);
        assert_eq!(report.physical_safe_completions, 0);
        assert_eq!(report.successful_but_unsafe, 1);
        assert_eq!(report.safety_violation_rows, 1);
    }

    #[test]
    fn unknown_safety_or_world_state_fails_closed() {
        let mut trial = valid_trial("trial-01");
        trial.safety = SafetyEvidence::Unknown;
        let mut second = valid_trial("trial-02");
        second.world_state = WorldStateContinuity::Unknown;

        let report = audit_trials(&[trial, second], None);
        assert_eq!(report.physical_safe_completions, 0);
        assert_eq!(report.infrastructure_indeterminate, 2);
    }

    #[test]
    fn safe_hold_and_human_handoff_are_not_autonomous_completions() {
        let mut hold = valid_trial("trial-01");
        hold.terminal = TerminalOutcome::SafeHold;
        hold.intervention = InterventionKind::SafeHold;
        let mut handoff = valid_trial("trial-02");
        handoff.terminal = TerminalOutcome::HumanHandoff;
        handoff.intervention = InterventionKind::HumanHandoff;

        let report = audit_trials(&[hold, handoff], None);
        assert_eq!(report.physical_safe_completions, 0);
        assert_eq!(report.safe_holds, 1);
        assert_eq!(report.human_handoffs, 1);
    }

    #[test]
    fn controller_rollback_requires_preserved_world_and_executed_actions() {
        let mut rollback = valid_trial("trial-01");
        rollback.intervention = InterventionKind::ControllerRollback;
        rollback.executed_action_count = 0;
        let mut executed = valid_trial("trial-02");
        executed.intervention = InterventionKind::ControllerRollback;

        let report = audit_trials(&[rollback, executed], None);
        assert_eq!(report.physical_safe_completions, 1);
        assert_eq!(report.infrastructure_indeterminate, 1);
    }

    #[test]
    fn every_row_with_a_duplicate_trial_id_is_excluded() {
        let first = valid_trial("duplicate");
        let mut second = valid_trial("duplicate");
        second.arm_id = "assurance-sidecar".to_owned();

        let report = audit_trials(&[first, second], None);
        assert_eq!(report.duplicate_trial_ids, vec!["duplicate".to_owned()]);
        assert_eq!(report.duplicate_trial_rows, 2);
        assert_eq!(report.invalid_identity_rows, 2);
        assert_eq!(report.valid_confirmatory_rows, 0);
        assert_eq!(report.physical_safe_completions, 0);
    }

    #[test]
    fn expected_manifest_exposes_missing_and_unplanned_trials() {
        let observed = vec![valid_trial("expected-01"), valid_trial("extra-01")];
        let expected = vec!["expected-01".to_owned(), "missing-01".to_owned()];

        let report = audit_trials(&observed, Some(&expected));
        assert_eq!(report.expected_trial_count, Some(2));
        assert_eq!(report.missing_expected_trial_ids, vec!["missing-01".to_owned()]);
        assert_eq!(report.unplanned_rows, 1);
        assert_eq!(report.valid_confirmatory_rows, 1);
    }

    #[test]
    fn blank_identity_fields_are_rejected() {
        let mut trial = valid_trial("trial-01");
        trial.evaluator_digest = "  ".to_owned();

        let report = audit_trials(&[trial], None);
        assert_eq!(report.invalid_identity_rows, 1);
        assert_eq!(report.valid_confirmatory_rows, 0);
        assert_eq!(report.physical_safe_completions, 0);
    }

    #[test]
    fn duplicate_expected_manifest_invalidates_confirmatory_accounting() {
        let observed = vec![valid_trial("trial-01")];
        let expected = vec!["trial-01".to_owned(), "trial-01".to_owned()];

        let report = audit_trials(&observed, Some(&expected));
        assert_eq!(report.expected_manifest_valid, Some(false));
        assert_eq!(
            report.duplicate_expected_trial_ids,
            vec!["trial-01".to_owned()]
        );
        assert_eq!(report.valid_confirmatory_rows, 0);
        assert_eq!(report.invalid_manifest_rows, 1);
    }

    #[test]
    fn blank_expected_manifest_id_invalidates_confirmatory_accounting() {
        let observed = vec![valid_trial("trial-01")];
        let expected = vec!["trial-01".to_owned(), "   ".to_owned()];

        let report = audit_trials(&observed, Some(&expected));
        assert_eq!(report.expected_manifest_valid, Some(false));
        assert_eq!(report.blank_expected_trial_ids, 1);
        assert_eq!(report.valid_confirmatory_rows, 0);
    }

    #[test]
    fn raw_safety_and_rewind_markers_survive_duplicate_row_exclusion() {
        let mut rewind = valid_trial("duplicate");
        rewind.intervention = InterventionKind::SimulatorStateRewind;
        rewind.world_state = WorldStateContinuity::SimulatorRewound;

        let mut unsafe_row = valid_trial("duplicate");
        unsafe_row.safety = SafetyEvidence::ViolationObserved;

        let report = audit_trials(&[rewind, unsafe_row], None);
        assert_eq!(report.valid_confirmatory_rows, 0);
        assert_eq!(report.invalid_identity_rows, 2);
        assert_eq!(report.simulator_rewind_rows, 1);
        assert_eq!(report.safety_violation_rows, 1);
    }

}
