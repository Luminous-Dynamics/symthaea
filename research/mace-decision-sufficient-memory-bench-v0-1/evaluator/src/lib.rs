// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-free scoring core for MACE Decision-Sufficient Memory (DSM).
//!
//! Security boundary:
//! - Candidate-facing JSONL contains only candidate episode data.
//! - EvaluatorCaseV1 contains hidden losses and adjudication labels. It must
//!   be constructed and retained only by the evaluator process.
//! - Never serialize EvaluatorCaseV1 into a candidate prompt, request,
//!   artifact, log, or shared fixture.
//!
//! This crate scores already-produced candidate outputs. It does not parse the
//! public JSONL format, generate worlds, run a memory backend, or establish
//! production security. Those are separate adapter/evaluator responsibilities.

use std::collections::BTreeSet;

/// Evaluator-owned action loss, expressed in integer micro-loss units.
///
/// Integer losses avoid platform-dependent floating-point tie handling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionLossV1 {
    pub action: String,
    pub loss_micros: u64,
}

/// Hidden evaluator data. Never expose this structure to candidate systems.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluatorCaseV1 {
    pub episode_id: String,
    pub allowed_actions: Vec<String>,
    pub action_losses: Vec<ActionLossV1>,
    pub no_memory_action: String,
    pub oracle_action: String,
    /// Evidence IDs independently adjudicated as admissible at query time.
    pub admissible_evidence_ids: Vec<String>,
    /// Evidence required to achieve the reference evidence objective.
    pub required_evidence_ids: Vec<String>,
}

/// Candidate-facing output from one memory system for one episode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateRunV1 {
    pub episode_id: String,
    pub system_id: String,
    pub chosen_action: String,
    /// Ordered evidence IDs returned by Search / evidence resolution.
    pub retrieved_evidence_ids: Vec<String>,
}

/// Exact integer ratio. A missing ratio is used when its denominator is zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CountRatioV1 {
    pub numerator: u64,
    pub denominator: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunScoreV1 {
    pub system_id: String,
    pub chosen_action: String,
    pub decision_loss_micros: u64,
    pub oracle_loss_micros: u64,
    pub decision_regret_micros: u64,
    pub decision_optimal: bool,
    pub returned_evidence_count: u64,
    pub admissible_returned_evidence_count: u64,
    pub inadmissible_returned_evidence_count: u64,
    pub required_evidence_recall: Option<CountRatioV1>,
    pub admissible_evidence_precision: Option<CountRatioV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedScoreV1 {
    pub episode_id: String,
    pub memory: RunScoreV1,
    pub no_memory: RunScoreV1,
    /// Positive means memory lowered decision loss; negative means memory
    /// increased decision loss relative to the paired no-memory control.
    pub memory_gain_micros: i128,
    pub memory_induced_harm: bool,
    pub memory_changed_action: bool,
    pub changed_action_helped: bool,
    pub changed_action_harmed: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseTimingsV1 {
    pub add_ingest_ns: u64,
    pub index_update_ns: u64,
    pub retrieval_ns: u64,
    pub evidence_verification_ns: u64,
    pub policy_application_ns: u64,
    pub maintenance_ns: u64,
    pub unattributed_ns: u64,
    pub wall_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CostReceiptV1 {
    pub phases: PhaseTimingsV1,
    /// Monotonic, measured resource cost in an implementation-defined unit.
    pub measured_cost_microunits: Option<u64>,
    /// Separately identified estimate; never substitute this for measurement.
    pub estimated_cost_microunits: Option<u64>,
    pub retained_bytes: u64,
    pub index_bytes: u64,
    pub network_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostReceiptErrorV1 {
    PhaseSumOverflow,
    PhasesExceedWallTime,
}

impl CostReceiptV1 {
    /// Ensure non-overlapping measured phases do not exceed total wall time.
    pub fn validate(&self) -> Result<(), CostReceiptErrorV1> {
        let p = self.phases;
        let sum = [
            p.add_ingest_ns,
            p.index_update_ns,
            p.retrieval_ns,
            p.evidence_verification_ns,
            p.policy_application_ns,
            p.maintenance_ns,
            p.unattributed_ns,
        ]
        .into_iter()
        .try_fold(0_u128, |acc, value| {
            acc.checked_add(u128::from(value))
        })
        .ok_or(CostReceiptErrorV1::PhaseSumOverflow)?;

        if sum > u128::from(p.wall_ns) {
            return Err(CostReceiptErrorV1::PhasesExceedWallTime);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationErrorV1 {
    EmptyEpisodeId,
    EmptySystemId,
    DuplicateAllowedAction,
    DuplicateLossAction,
    LossActionSetMismatch,
    InvalidNoMemoryAction,
    InvalidOracleAction,
    OracleActionIsNotOptimal,
    DuplicateAdmissibleEvidenceId,
    DuplicateRequiredEvidenceId,
    RequiredEvidenceIsNotAdmissible,
    EpisodeMismatch,
    SameSystemUsedForBothArms,
    InvalidCandidateAction,
    DuplicateReturnedEvidenceId,
    NoMemoryArmContainsEvidence,
    NoMemoryArmActionMismatch,
    MissingActionLoss,
}

fn contains(values: &[String], target: &str) -> bool {
    values.iter().any(|value| value == target)
}

fn unique(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
}

fn action_loss(case: &EvaluatorCaseV1, action: &str) -> Option<u64> {
    case.action_losses
        .iter()
        .find(|entry| entry.action == action)
        .map(|entry| entry.loss_micros)
}

/// Validate evaluator-only ground truth before scoring any candidate output.
pub fn validate_case(case: &EvaluatorCaseV1) -> Result<(), EvaluationErrorV1> {
    if case.episode_id.is_empty() {
        return Err(EvaluationErrorV1::EmptyEpisodeId);
    }
    if case.allowed_actions.is_empty() {
        return Err(EvaluationErrorV1::LossActionSetMismatch);
    }
    if !unique(&case.allowed_actions) {
        return Err(EvaluationErrorV1::DuplicateAllowedAction);
    }

    let loss_actions: Vec<String> = case
        .action_losses
        .iter()
        .map(|entry| entry.action.clone())
        .collect();
    if !unique(&loss_actions) {
        return Err(EvaluationErrorV1::DuplicateLossAction);
    }

    let allowed: BTreeSet<&str> = case.allowed_actions.iter().map(String::as_str).collect();
    let losses: BTreeSet<&str> = case
        .action_losses
        .iter()
        .map(|entry| entry.action.as_str())
        .collect();
    if allowed != losses {
        return Err(EvaluationErrorV1::LossActionSetMismatch);
    }

    if !contains(&case.allowed_actions, &case.no_memory_action) {
        return Err(EvaluationErrorV1::InvalidNoMemoryAction);
    }
    if !contains(&case.allowed_actions, &case.oracle_action) {
        return Err(EvaluationErrorV1::InvalidOracleAction);
    }

    let oracle_loss =
        action_loss(case, &case.oracle_action).ok_or(EvaluationErrorV1::MissingActionLoss)?;
    let minimum_loss = case
        .action_losses
        .iter()
        .map(|entry| entry.loss_micros)
        .min()
        .ok_or(EvaluationErrorV1::LossActionSetMismatch)?;
    if oracle_loss != minimum_loss {
        return Err(EvaluationErrorV1::OracleActionIsNotOptimal);
    }

    if !unique(&case.admissible_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateAdmissibleEvidenceId);
    }
    if !unique(&case.required_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateRequiredEvidenceId);
    }
    if case
        .required_evidence_ids
        .iter()
        .any(|id| !contains(&case.admissible_evidence_ids, id))
    {
        return Err(EvaluationErrorV1::RequiredEvidenceIsNotAdmissible);
    }
    Ok(())
}

fn validate_candidate(
    case: &EvaluatorCaseV1,
    candidate: &CandidateRunV1,
) -> Result<(), EvaluationErrorV1> {
    if candidate.episode_id != case.episode_id {
        return Err(EvaluationErrorV1::EpisodeMismatch);
    }
    if candidate.system_id.is_empty() {
        return Err(EvaluationErrorV1::EmptySystemId);
    }
    if !contains(&case.allowed_actions, &candidate.chosen_action) {
        return Err(EvaluationErrorV1::InvalidCandidateAction);
    }
    if !unique(&candidate.retrieved_evidence_ids) {
        return Err(EvaluationErrorV1::DuplicateReturnedEvidenceId);
    }
    Ok(())
}

fn score_run(
    case: &EvaluatorCaseV1,
    candidate: &CandidateRunV1,
) -> Result<RunScoreV1, EvaluationErrorV1> {
    validate_candidate(case, candidate)?;

    let decision_loss_micros = action_loss(case, &candidate.chosen_action)
        .ok_or(EvaluationErrorV1::MissingActionLoss)?;
    let oracle_loss_micros =
        action_loss(case, &case.oracle_action).ok_or(EvaluationErrorV1::MissingActionLoss)?;
    let admissible: BTreeSet<&str> = case
        .admissible_evidence_ids
        .iter()
        .map(String::as_str)
        .collect();
    let required: BTreeSet<&str> = case
        .required_evidence_ids
        .iter()
        .map(String::as_str)
        .collect();

    let returned = candidate.retrieved_evidence_ids.len() as u64;
    let admissible_returned = candidate
        .retrieved_evidence_ids
        .iter()
        .filter(|id| admissible.contains(id.as_str()))
        .count() as u64;
    let required_recalled = candidate
        .retrieved_evidence_ids
        .iter()
        .filter(|id| required.contains(id.as_str()))
        .count() as u64;

    Ok(RunScoreV1 {
        system_id: candidate.system_id.clone(),
        chosen_action: candidate.chosen_action.clone(),
        decision_loss_micros,
        oracle_loss_micros,
        decision_regret_micros: decision_loss_micros - oracle_loss_micros,
        decision_optimal: decision_loss_micros == oracle_loss_micros,
        returned_evidence_count: returned,
        admissible_returned_evidence_count: admissible_returned,
        inadmissible_returned_evidence_count: returned - admissible_returned,
        required_evidence_recall: (!required.is_empty()).then_some(CountRatioV1 {
            numerator: required_recalled,
            denominator: required.len() as u64,
        }),
        admissible_evidence_precision: (returned > 0).then_some(CountRatioV1 {
            numerator: admissible_returned,
            denominator: returned,
        }),
    })
}

/// Score memory and no-memory arms on the exact same evaluator episode.
///
/// The no-memory arm is constrained to the declared no-memory action and must
/// not return evidence. Candidate outputs are validated but never used to
/// derive evaluator labels, losses, admissible IDs or the oracle action.
pub fn score_paired(
    case: &EvaluatorCaseV1,
    memory: &CandidateRunV1,
    no_memory: &CandidateRunV1,
) -> Result<PairedScoreV1, EvaluationErrorV1> {
    validate_case(case)?;
    validate_candidate(case, memory)?;
    validate_candidate(case, no_memory)?;

    if memory.system_id == no_memory.system_id {
        return Err(EvaluationErrorV1::SameSystemUsedForBothArms);
    }
    if no_memory.chosen_action != case.no_memory_action {
        return Err(EvaluationErrorV1::NoMemoryArmActionMismatch);
    }
    if !no_memory.retrieved_evidence_ids.is_empty() {
        return Err(EvaluationErrorV1::NoMemoryArmContainsEvidence);
    }

    let memory_score = score_run(case, memory)?;
    let no_memory_score = score_run(case, no_memory)?;
    let memory_gain_micros =
        i128::from(no_memory_score.decision_loss_micros)
            - i128::from(memory_score.decision_loss_micros);
    let memory_changed_action = memory.chosen_action != no_memory.chosen_action;

    Ok(PairedScoreV1 {
        episode_id: case.episode_id.clone(),
        memory: memory_score,
        no_memory: no_memory_score,
        memory_gain_micros,
        memory_induced_harm: memory_gain_micros < 0,
        memory_changed_action,
        changed_action_helped: memory_changed_action && memory_gain_micros > 0,
        changed_action_harmed: memory_changed_action && memory_gain_micros < 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Public known-answer fixture only. Confirmatory evaluator answer keys and
    // held-out truth must live outside the candidate-accessible repository.
    fn case() -> EvaluatorCaseV1 {
        EvaluatorCaseV1 {
            episode_id: "known-answer/0001".to_owned(),
            allowed_actions: vec![
                "approve".to_owned(),
                "hold".to_owned(),
                "abstain".to_owned(),
            ],
            action_losses: vec![
                ActionLossV1 {
                    action: "approve".to_owned(),
                    loss_micros: 9_000_000,
                },
                ActionLossV1 {
                    action: "hold".to_owned(),
                    loss_micros: 0,
                },
                ActionLossV1 {
                    action: "abstain".to_owned(),
                    loss_micros: 2_000_000,
                },
            ],
            no_memory_action: "abstain".to_owned(),
            oracle_action: "hold".to_owned(),
            admissible_evidence_ids: vec!["signed-current".to_owned()],
            required_evidence_ids: vec!["signed-current".to_owned()],
        }
    }

    fn run(system: &str, action: &str, evidence: &[&str]) -> CandidateRunV1 {
        CandidateRunV1 {
            episode_id: "known-answer/0001".to_owned(),
            system_id: system.to_owned(),
            chosen_action: action.to_owned(),
            retrieved_evidence_ids: evidence.iter().map(|id| (*id).to_owned()).collect(),
        }
    }

    #[test]
    fn qualified_memory_reduces_regret_against_paired_no_memory() {
        let result = score_paired(
            &case(),
            &run("mace-qualified", "hold", &["signed-current"]),
            &run("no-memory", "abstain", &[]),
        )
        .expect("valid paired evaluation");

        assert_eq!(result.memory.decision_regret_micros, 0);
        assert_eq!(result.no_memory.decision_regret_micros, 2_000_000);
        assert_eq!(result.memory_gain_micros, 2_000_000);
        assert!(!result.memory_induced_harm);
        assert!(result.changed_action_helped);
        assert_eq!(
            result.memory.admissible_evidence_precision,
            Some(CountRatioV1 {
                numerator: 1,
                denominator: 1,
            })
        );
    }

    #[test]
    fn stale_or_unknown_evidence_is_counted_as_inadmissible() {
        let result = score_paired(
            &case(),
            &run("mace-stale", "approve", &["expired-account"]),
            &run("no-memory", "abstain", &[]),
        )
        .expect("candidate errors are scored, not silently trusted");

        assert_eq!(result.memory.inadmissible_returned_evidence_count, 1);
        assert_eq!(
            result.memory.admissible_evidence_precision,
            Some(CountRatioV1 {
                numerator: 0,
                denominator: 1,
            })
        );
        assert_eq!(result.memory_gain_micros, -7_000_000);
        assert!(result.memory_induced_harm);
        assert!(result.changed_action_harmed);
    }

    #[test]
    fn candidate_cannot_change_episode_identity_between_arms() {
        let mut memory = run("mace", "hold", &["signed-current"]);
        memory.episode_id = "different-world".to_owned();
        let error = score_paired(&case(), &memory, &run("no-memory", "abstain", &[]));
        assert_eq!(error, Err(EvaluationErrorV1::EpisodeMismatch));
    }

    #[test]
    fn no_memory_control_must_not_return_evidence() {
        let error = score_paired(
            &case(),
            &run("mace", "hold", &["signed-current"]),
            &run("no-memory", "abstain", &["signed-current"]),
        );
        assert_eq!(error, Err(EvaluationErrorV1::NoMemoryArmContainsEvidence));
    }

    #[test]
    fn oracle_action_must_be_a_minimum_loss_action() {
        let mut invalid = case();
        invalid.oracle_action = "approve".to_owned();
        assert_eq!(
            validate_case(&invalid),
            Err(EvaluationErrorV1::OracleActionIsNotOptimal)
        );
    }

    #[test]
    fn duplicate_candidate_evidence_ids_are_rejected() {
        let result = score_paired(
            &case(),
            &run("mace", "hold", &["signed-current", "signed-current"]),
            &run("no-memory", "abstain", &[]),
        );
        assert_eq!(result, Err(EvaluationErrorV1::DuplicateReturnedEvidenceId));
    }

    #[test]
    fn empty_retrieval_has_no_precision_value_not_a_perfect_precision_claim() {
        let score = score_run(&case(), &run("empty", "abstain", &[]))
            .expect("empty retrieval is a valid candidate run");
        assert_eq!(score.admissible_evidence_precision, None);
        assert_eq!(
            score.required_evidence_recall,
            Some(CountRatioV1 {
                numerator: 0,
                denominator: 1,
            })
        );
    }

    #[test]
    fn phase_cost_receipt_rejects_overstated_phase_time() {
        let receipt = CostReceiptV1 {
            phases: PhaseTimingsV1 {
                add_ingest_ns: 30,
                index_update_ns: 30,
                retrieval_ns: 30,
                evidence_verification_ns: 30,
                policy_application_ns: 30,
                maintenance_ns: 0,
                unattributed_ns: 0,
                wall_ns: 100,
            },
            measured_cost_microunits: Some(10),
            estimated_cost_microunits: None,
            retained_bytes: 100,
            index_bytes: 40,
            network_bytes: 20,
        };
        assert_eq!(
            receipt.validate(),
            Err(CostReceiptErrorV1::PhasesExceedWallTime)
        );
    }

    #[test]
    fn phase_cost_receipt_accepts_reconciled_wall_time() {
        let receipt = CostReceiptV1 {
            phases: PhaseTimingsV1 {
                add_ingest_ns: 20,
                index_update_ns: 10,
                retrieval_ns: 15,
                evidence_verification_ns: 20,
                policy_application_ns: 10,
                maintenance_ns: 5,
                unattributed_ns: 20,
                wall_ns: 100,
            },
            measured_cost_microunits: None,
            estimated_cost_microunits: Some(12),
            retained_bytes: 100,
            index_bytes: 40,
            network_bytes: 20,
        };
        assert_eq!(receipt.validate(), Ok(()));
    }
}
