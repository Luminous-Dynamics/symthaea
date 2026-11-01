use std::collections::BTreeMap;

use crate::{ResourceVector, SelectionReceipt};

/// A statement covers exactly one resource dimension; no implicit scalarization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OnlineCostDimension {
    Tokens,
    ToolCalls,
    CpuMs,
    GpuMs,
    WallMs,
    MicroUsd,
}

impl OnlineCostDimension {
    fn value(self, value: ResourceVector) -> u128 {
        match self {
            Self::Tokens => value.tokens,
            Self::ToolCalls => value.tool_calls,
            Self::CpuMs => value.cpu_ms,
            Self::GpuMs => value.gpu_ms,
            Self::WallMs => value.wall_ms,
            Self::MicroUsd => value.micro_usd,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnlinePredictionStatement {
    pub statement_id: u64,
    pub selection: SelectionReceipt,
    pub dimension: OnlineCostDimension,
    pub lower_bound: u128,
    pub upper_bound: u128,
    pub issued_at_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineCompetingTerminal {
    PermissionRevoked,
    UserAbandoned,
    TaskAborted,
    ExternalInputUnavailable,
}

/// Unknown is an interim state. One later final receipt may reconcile it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineFeedbackDisposition {
    ActionOutcomeUnknown { action_id_digest: [u8; 32] },
    ExactVerifiedCost {
        observed_cost: u128,
        meter_receipt_digest: [u8; 32],
        completion_verifier_digest: [u8; 32],
    },
    CoverageBitOnly {
        covered: bool,
        meter_receipt_digest: [u8; 32],
        evaluator_receipt_digest: [u8; 32],
    },
    RightCensored {
        observed_lower_bound: u128,
        meter_receipt_digest: [u8; 32],
    },
    CompetingTerminal {
        terminal: OnlineCompetingTerminal,
        evaluator_receipt_digest: [u8; 32],
    },
}

impl OnlineFeedbackDisposition {
    fn is_interim(self) -> bool {
        matches!(self, Self::ActionOutcomeUnknown { .. })
    }

    fn validate(self) -> Result<(), OnlineSelectionError> {
        match self {
            Self::ActionOutcomeUnknown { action_id_digest } => {
                if is_zero(action_id_digest) {
                    return Err(OnlineSelectionError::MissingActionIdentity);
                }
            }
            Self::ExactVerifiedCost {
                meter_receipt_digest,
                completion_verifier_digest,
                ..
            } => {
                if is_zero(meter_receipt_digest) {
                    return Err(OnlineSelectionError::MissingMeterReceipt);
                }
                if is_zero(completion_verifier_digest) {
                    return Err(OnlineSelectionError::MissingCompletionVerifier);
                }
            }
            Self::CoverageBitOnly {
                meter_receipt_digest,
                evaluator_receipt_digest,
                ..
            } => {
                if is_zero(meter_receipt_digest) {
                    return Err(OnlineSelectionError::MissingMeterReceipt);
                }
                if is_zero(evaluator_receipt_digest) {
                    return Err(OnlineSelectionError::MissingEvaluatorReceipt);
                }
            }
            Self::RightCensored { meter_receipt_digest, .. } => {
                if is_zero(meter_receipt_digest) {
                    return Err(OnlineSelectionError::MissingMeterReceipt);
                }
            }
            Self::CompetingTerminal { evaluator_receipt_digest, .. } => {
                if is_zero(evaluator_receipt_digest) {
                    return Err(OnlineSelectionError::MissingEvaluatorReceipt);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnlineFeedbackReceipt {
    pub receipt_sequence: u64,
    pub statement_id: u64,
    pub received_at_tick: u64,
    pub disposition: OnlineFeedbackDisposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineSelectionError {
    InvalidStatementId,
    NonMonotoneStatementId,
    InvalidEpisodeId,
    InvalidDecisionSequence,
    NonMonotoneEpisodeDecision,
    MissingCandidatePoolDigest,
    MissingSelectionPolicyDigest,
    MissingSelectedPopulationDigest,
    MissingPreStateDigest,
    MissingFrontierDigest,
    MissingSelectedOperationId,
    InvalidPredictionInterval,
    StatementNotFound,
    InvalidFeedbackSequence,
    NonMonotoneFeedbackTick,
    FeedbackBeforeStatement,
    DuplicateFinalFeedback,
    RepeatedUnknownFeedback,
    MissingActionIdentity,
    MissingMeterReceipt,
    MissingCompletionVerifier,
    MissingEvaluatorReceipt,
}

impl std::fmt::Display for OnlineSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for OnlineSelectionError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OnlineSelectionLedger {
    statements: BTreeMap<u64, OnlinePredictionStatement>,
    feedback: BTreeMap<u64, Vec<OnlineFeedbackReceipt>>,
    last_statement_id: u64,
    last_feedback_sequence: u64,
    last_feedback_tick: u64,
    last_episode_decision: BTreeMap<u64, u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnlineFcrReport {
    pub as_of_tick: u64,
    pub issued_selected_statements: usize,
    pub resolved_coverage_statements: usize,
    pub coverage_hits: usize,
    pub coverage_misses: usize,
    /// Observed misses divided by resolved coverage statements. This is
    /// descriptive accounting, not selection-corrected FCR or a guarantee.
    pub observed_fcr: Option<f64>,
    pub pending_feedback: usize,
    pub action_outcome_unknown: usize,
    pub right_censored_unresolved: usize,
    pub competing_terminal: usize,
}

impl OnlineSelectionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append_statement(
        &mut self,
        statement: OnlinePredictionStatement,
    ) -> Result<(), OnlineSelectionError> {
        validate_statement(&statement)?;
        if statement.statement_id <= self.last_statement_id {
            return Err(OnlineSelectionError::NonMonotoneStatementId);
        }
        let prior_decision = self
            .last_episode_decision
            .get(&statement.selection.episode_id)
            .copied()
            .unwrap_or(0);
        if statement.selection.decision_sequence <= prior_decision {
            return Err(OnlineSelectionError::NonMonotoneEpisodeDecision);
        }
        self.last_statement_id = statement.statement_id;
        self.last_episode_decision.insert(
            statement.selection.episode_id,
            statement.selection.decision_sequence,
        );
        self.statements.insert(statement.statement_id, statement);
        Ok(())
    }

    pub fn append_feedback(
        &mut self,
        receipt: OnlineFeedbackReceipt,
    ) -> Result<(), OnlineSelectionError> {
        let statement = self
            .statements
            .get(&receipt.statement_id)
            .ok_or(OnlineSelectionError::StatementNotFound)?;
        if receipt.receipt_sequence != self.last_feedback_sequence + 1 {
            return Err(OnlineSelectionError::InvalidFeedbackSequence);
        }
        if receipt.received_at_tick < self.last_feedback_tick {
            return Err(OnlineSelectionError::NonMonotoneFeedbackTick);
        }
        if receipt.received_at_tick < statement.issued_at_tick {
            return Err(OnlineSelectionError::FeedbackBeforeStatement);
        }
        receipt.disposition.validate()?;

        if let Some(previous) = self
            .feedback
            .get(&receipt.statement_id)
            .and_then(|items| items.last())
        {
            if !previous.disposition.is_interim() {
                return Err(OnlineSelectionError::DuplicateFinalFeedback);
            }
            if receipt.disposition.is_interim() {
                return Err(OnlineSelectionError::RepeatedUnknownFeedback);
            }
        }

        self.last_feedback_sequence = receipt.receipt_sequence;
        self.last_feedback_tick = receipt.received_at_tick;
        self.feedback
            .entry(receipt.statement_id)
            .or_default()
            .push(receipt);
        Ok(())
    }

    /// Replay-prefix report: only statements and feedback observable at this
    /// logical tick are included. Censoring and competing terminals are never
    /// silently relabelled as ordinary covered proof completions.
    pub fn report(&self, as_of_tick: u64) -> OnlineFcrReport {
        let mut report = OnlineFcrReport {
            as_of_tick,
            issued_selected_statements: 0,
            resolved_coverage_statements: 0,
            coverage_hits: 0,
            coverage_misses: 0,
            observed_fcr: None,
            pending_feedback: 0,
            action_outcome_unknown: 0,
            right_censored_unresolved: 0,
            competing_terminal: 0,
        };

        for (statement_id, statement) in &self.statements {
            if statement.issued_at_tick > as_of_tick {
                continue;
            }
            report.issued_selected_statements += 1;
            let latest = self.feedback.get(statement_id).and_then(|items| {
                items
                    .iter()
                    .rev()
                    .find(|item| item.received_at_tick <= as_of_tick)
            });
            let Some(receipt) = latest else {
                report.pending_feedback += 1;
                continue;
            };

            match receipt.disposition {
                OnlineFeedbackDisposition::ActionOutcomeUnknown { .. } => {
                    report.action_outcome_unknown += 1;
                }
                OnlineFeedbackDisposition::ExactVerifiedCost { observed_cost, .. } => {
                    record_coverage(
                        &mut report,
                        observed_cost >= statement.lower_bound
                            && observed_cost <= statement.upper_bound,
                    );
                }
                OnlineFeedbackDisposition::CoverageBitOnly { covered, .. } => {
                    record_coverage(&mut report, covered);
                }
                OnlineFeedbackDisposition::RightCensored {
                    observed_lower_bound,
                    ..
                } => {
                    if observed_lower_bound > statement.upper_bound {
                        record_coverage(&mut report, false);
                    } else {
                        report.right_censored_unresolved += 1;
                    }
                }
                OnlineFeedbackDisposition::CompetingTerminal { .. } => {
                    report.competing_terminal += 1;
                }
            }
        }
        if report.resolved_coverage_statements > 0 {
            report.observed_fcr = Some(
                report.coverage_misses as f64 / report.resolved_coverage_statements as f64,
            );
        }
        report
    }

    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    pub fn feedback_count(&self) -> usize {
        self.feedback.values().map(Vec::len).sum()
    }
}

fn validate_statement(
    statement: &OnlinePredictionStatement,
) -> Result<(), OnlineSelectionError> {
    if statement.statement_id == 0 {
        return Err(OnlineSelectionError::InvalidStatementId);
    }
    if statement.selection.episode_id == 0 {
        return Err(OnlineSelectionError::InvalidEpisodeId);
    }
    if statement.selection.decision_sequence == 0 {
        return Err(OnlineSelectionError::InvalidDecisionSequence);
    }
    if is_zero(statement.selection.candidate_pool_digest) {
        return Err(OnlineSelectionError::MissingCandidatePoolDigest);
    }
    if is_zero(statement.selection.selection_policy_digest) {
        return Err(OnlineSelectionError::MissingSelectionPolicyDigest);
    }
    if is_zero(statement.selection.selected_population_digest) {
        return Err(OnlineSelectionError::MissingSelectedPopulationDigest);
    }
    if is_zero(statement.selection.pre_state_digest) {
        return Err(OnlineSelectionError::MissingPreStateDigest);
    }
    if is_zero(statement.selection.frontier_digest) {
        return Err(OnlineSelectionError::MissingFrontierDigest);
    }
    if statement.selection.selected_operation_id.trim().is_empty() {
        return Err(OnlineSelectionError::MissingSelectedOperationId);
    }
    if statement.lower_bound > statement.upper_bound {
        return Err(OnlineSelectionError::InvalidPredictionInterval);
    }
    Ok(())
}

fn record_coverage(report: &mut OnlineFcrReport, covered: bool) {
    report.resolved_coverage_statements += 1;
    if covered {
        report.coverage_hits += 1;
    } else {
        report.coverage_misses += 1;
    }
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statement(id: u64, episode: u64, decision: u64, tick: u64) -> OnlinePredictionStatement {
        OnlinePredictionStatement {
            statement_id: id,
            selection: SelectionReceipt {
                episode_id: episode,
                decision_sequence: decision,
                candidate_pool_digest: [1; 32],
                selection_policy_digest: [2; 32],
                selected_population_digest: [3; 32],
                pre_state_digest: [4; 32],
                frontier_digest: [5; 32],
                selected_operation_id: format!("op-{id}"),
                selected_operation_family: 7,
            },
            dimension: OnlineCostDimension::Tokens,
            lower_bound: 10,
            upper_bound: 20,
            issued_at_tick: tick,
        }
    }

    fn receipt(
        sequence: u64,
        id: u64,
        tick: u64,
        disposition: OnlineFeedbackDisposition,
    ) -> OnlineFeedbackReceipt {
        OnlineFeedbackReceipt {
            receipt_sequence: sequence,
            statement_id: id,
            received_at_tick: tick,
            disposition,
        }
    }

    fn exact(value: u128) -> OnlineFeedbackDisposition {
        OnlineFeedbackDisposition::ExactVerifiedCost {
            observed_cost: value,
            meter_receipt_digest: [6; 32],
            completion_verifier_digest: [7; 32],
        }
    }

    fn bit(covered: bool) -> OnlineFeedbackDisposition {
        OnlineFeedbackDisposition::CoverageBitOnly {
            covered,
            meter_receipt_digest: [6; 32],
            evaluator_receipt_digest: [7; 32],
        }
    }

    #[test]
    fn exact_verified_labels_calculate_selected_statement_fcr() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(1, 10, 1, 1)).unwrap();
        ledger.append_statement(statement(2, 11, 1, 2)).unwrap();
        ledger.append_feedback(receipt(1, 1, 3, exact(15))).unwrap();
        ledger.append_feedback(receipt(2, 2, 4, exact(25))).unwrap();
        let report = ledger.report(4);
        assert_eq!(report.issued_selected_statements, 2);
        assert_eq!(report.resolved_coverage_statements, 2);
        assert_eq!(report.coverage_hits, 1);
        assert_eq!(report.coverage_misses, 1);
        assert_eq!(report.observed_fcr, Some(0.5));
    }

    #[test]
    fn delayed_feedback_is_absent_from_earlier_replay_prefixes() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(1, 10, 1, 1)).unwrap();
        ledger.append_feedback(receipt(1, 1, 8, bit(true))).unwrap();
        let before = ledger.report(7);
        assert_eq!(before.pending_feedback, 1);
        assert_eq!(before.resolved_coverage_statements, 0);
        let after = ledger.report(8);
        assert_eq!(after.pending_feedback, 0);
        assert_eq!(after.resolved_coverage_statements, 1);
        assert_eq!(after.observed_fcr, Some(0.0));
    }

    #[test]
    fn coverage_bit_feedback_scores_without_revealing_exact_cost() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(1, 10, 1, 1)).unwrap();
        ledger.append_feedback(receipt(1, 1, 2, bit(false))).unwrap();
        let report = ledger.report(2);
        assert_eq!(report.coverage_misses, 1);
        assert_eq!(report.observed_fcr, Some(1.0));
    }

    #[test]
    fn right_censoring_counts_only_definitive_interval_misses() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(1, 10, 1, 1)).unwrap();
        ledger.append_statement(statement(2, 11, 1, 1)).unwrap();
        ledger.append_feedback(receipt(1, 1, 2,
            OnlineFeedbackDisposition::RightCensored {
                observed_lower_bound: 21,
                meter_receipt_digest: [6; 32],
            },
        )).unwrap();
        ledger.append_feedback(receipt(2, 2, 3,
            OnlineFeedbackDisposition::RightCensored {
                observed_lower_bound: 15,
                meter_receipt_digest: [7; 32],
            },
        )).unwrap();
        let report = ledger.report(3);
        assert_eq!(report.resolved_coverage_statements, 1);
        assert_eq!(report.coverage_misses, 1);
        assert_eq!(report.right_censored_unresolved, 1);
        assert_eq!(report.observed_fcr, Some(1.0));
    }

    #[test]
    fn competing_terminal_is_neither_covered_nor_missed() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(1, 10, 1, 1)).unwrap();
        ledger.append_feedback(receipt(1, 1, 2,
            OnlineFeedbackDisposition::CompetingTerminal {
                terminal: OnlineCompetingTerminal::PermissionRevoked,
                evaluator_receipt_digest: [6; 32],
            },
        )).unwrap();
        let report = ledger.report(2);
        assert_eq!(report.competing_terminal, 1);
        assert_eq!(report.resolved_coverage_statements, 0);
        assert_eq!(report.observed_fcr, None);
    }

    #[test]
    fn unknown_action_outcome_requires_one_later_reconciliation() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(1, 10, 1, 1)).unwrap();
        ledger.append_feedback(receipt(1, 1, 2,
            OnlineFeedbackDisposition::ActionOutcomeUnknown {
                action_id_digest: [6; 32],
            },
        )).unwrap();
        assert_eq!(ledger.report(2).action_outcome_unknown, 1);
        assert_eq!(
            ledger.append_feedback(receipt(2, 1, 3,
                OnlineFeedbackDisposition::ActionOutcomeUnknown {
                    action_id_digest: [6; 32],
                },
            )),
            Err(OnlineSelectionError::RepeatedUnknownFeedback)
        );
        ledger.append_feedback(receipt(2, 1, 4, exact(12))).unwrap();
        assert_eq!(ledger.report(4).observed_fcr, Some(0.0));
    }

    #[test]
    fn statement_and_feedback_sequences_are_monotone_and_single_assignment() {
        let mut ledger = OnlineSelectionLedger::new();
        ledger.append_statement(statement(2, 10, 2, 1)).unwrap();
        assert_eq!(
            ledger.append_statement(statement(1, 11, 1, 1)),
            Err(OnlineSelectionError::NonMonotoneStatementId)
        );
        assert_eq!(
            ledger.append_statement(statement(3, 10, 2, 2)),
            Err(OnlineSelectionError::NonMonotoneEpisodeDecision)
        );
        ledger.append_statement(statement(3, 10, 3, 2)).unwrap();
        assert_eq!(
            ledger.append_feedback(receipt(2, 2, 3, bit(true))),
            Err(OnlineSelectionError::InvalidFeedbackSequence)
        );
        ledger.append_feedback(receipt(1, 2, 3, bit(true))).unwrap();
        assert_eq!(
            ledger.append_feedback(receipt(2, 2, 4, bit(true))),
            Err(OnlineSelectionError::DuplicateFinalFeedback)
        );
    }

    #[test]
    fn missing_receipts_and_feedback_before_statement_are_rejected() {
        let mut ledger = OnlineSelectionLedger::new();
        let mut invalid = statement(1, 10, 1, 4);
        invalid.selection.frontier_digest = [0; 32];
        assert_eq!(
            ledger.append_statement(invalid),
            Err(OnlineSelectionError::MissingFrontierDigest)
        );
        ledger.append_statement(statement(1, 10, 1, 4)).unwrap();
        assert_eq!(
            ledger.append_feedback(receipt(1, 1, 3, bit(true))),
            Err(OnlineSelectionError::FeedbackBeforeStatement)
        );
    }

    #[test]
    fn cost_dimensions_remain_separate() {
        let vector = ResourceVector::new(1, 2, 3, 4, 5, 6);
        assert_eq!(OnlineCostDimension::Tokens.value(vector), 1);
        assert_eq!(OnlineCostDimension::ToolCalls.value(vector), 2);
        assert_eq!(OnlineCostDimension::CpuMs.value(vector), 3);
        assert_eq!(OnlineCostDimension::GpuMs.value(vector), 4);
        assert_eq!(OnlineCostDimension::WallMs.value(vector), 5);
        assert_eq!(OnlineCostDimension::MicroUsd.value(vector), 6);
    }
}
