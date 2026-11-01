use std::collections::BTreeMap;

use crate::{ForecastError, RemainingCostForecast, ResourceVector, TerminalOutcome};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ForecastId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForecastReceipt {
    pub id: ForecastId,
    pub run_id: u64,
    pub decision_sequence: u64,
    pub task_family: u32,
    pub operation_family: u32,
    pub forecast: RemainingCostForecast,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForecastOutcomeReceipt {
    pub forecast_id: ForecastId,
    pub observed_sequence: u64,
    /// Realized work after the forecast decision. For right-censored episodes
    /// this is the observed lower bound, not the unknown eventual cost.
    pub observed_work: ResourceVector,
    pub terminal: TerminalOutcome,
    /// Digest of the evaluator's terminal-outcome record.
    pub evidence_digest: [u8; 32],
    /// Digest of the independent resource-meter receipt.
    pub meter_digest: [u8; 32],
    /// Present only when sufficient proof was independently verified.
    pub verifier_digest: Option<[u8; 32]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForecastJournalError {
    InvalidForecastId,
    InvalidRunId,
    InvalidDecisionSequence,
    DuplicateForecastId,
    DuplicateDecisionSequence,
    NonIncreasingDecisionSequence,
    InvalidForecast(ForecastError),
    UnknownForecast,
    OutcomeAlreadyRecorded,
    OutcomePrecedesForecast,
    MissingEvidenceDigest,
    MissingMeterDigest,
    MissingVerifierDigest,
    UnexpectedZeroVerifierDigest,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForecastJournal {
    forecasts: BTreeMap<ForecastId, ForecastReceipt>,
    outcomes: BTreeMap<ForecastId, ForecastOutcomeReceipt>,
    last_sequence_by_run: BTreeMap<u64, u64>,
    sequences_by_run: BTreeMap<(u64, u64), ForecastId>,
}

impl ForecastJournal {
    /// Append one pre-execution forecast. Stored receipts are exposed only by
    /// immutable reference; this journal has no update or delete operation.
    pub fn append_forecast(
        &mut self,
        receipt: ForecastReceipt,
    ) -> Result<(), ForecastJournalError> {
        if receipt.id.0 == 0 {
            return Err(ForecastJournalError::InvalidForecastId);
        }
        if receipt.run_id == 0 {
            return Err(ForecastJournalError::InvalidRunId);
        }
        if receipt.decision_sequence == 0 {
            return Err(ForecastJournalError::InvalidDecisionSequence);
        }
        receipt
            .forecast
            .validate()
            .map_err(ForecastJournalError::InvalidForecast)?;
        if self.forecasts.contains_key(&receipt.id) {
            return Err(ForecastJournalError::DuplicateForecastId);
        }
        let sequence_key = (receipt.run_id, receipt.decision_sequence);
        if self.sequences_by_run.contains_key(&sequence_key) {
            return Err(ForecastJournalError::DuplicateDecisionSequence);
        }
        if self
            .last_sequence_by_run
            .get(&receipt.run_id)
            .is_some_and(|last| receipt.decision_sequence <= *last)
        {
            return Err(ForecastJournalError::NonIncreasingDecisionSequence);
        }

        self.sequences_by_run.insert(sequence_key, receipt.id);
        self.last_sequence_by_run
            .insert(receipt.run_id, receipt.decision_sequence);
        self.forecasts.insert(receipt.id, receipt);
        Ok(())
    }

    /// Append a separately sourced terminal observation for a prior forecast.
    /// A censored observation is a lower bound; it is never re-labelled as a
    /// proof-completion failure. A forecast can receive exactly one terminal
    /// evaluation. Corrections require a new forecast / evaluation identity.
    pub fn append_outcome(
        &mut self,
        outcome: ForecastOutcomeReceipt,
    ) -> Result<(), ForecastJournalError> {
        let forecast = self
            .forecasts
            .get(&outcome.forecast_id)
            .ok_or(ForecastJournalError::UnknownForecast)?;
        if self.outcomes.contains_key(&outcome.forecast_id) {
            return Err(ForecastJournalError::OutcomeAlreadyRecorded);
        }
        if outcome.observed_sequence <= forecast.decision_sequence {
            return Err(ForecastJournalError::OutcomePrecedesForecast);
        }
        if outcome.evidence_digest.iter().all(|byte| *byte == 0) {
            return Err(ForecastJournalError::MissingEvidenceDigest);
        }
        if outcome.meter_digest.iter().all(|byte| *byte == 0) {
            return Err(ForecastJournalError::MissingMeterDigest);
        }

        match (outcome.terminal, outcome.verifier_digest) {
            (TerminalOutcome::SufficientProof, None) => {
                return Err(ForecastJournalError::MissingVerifierDigest);
            }
            (TerminalOutcome::SufficientProof, Some(digest))
                if digest.iter().all(|byte| *byte == 0) =>
            {
                return Err(ForecastJournalError::UnexpectedZeroVerifierDigest);
            }
            (_, Some(digest)) if digest.iter().all(|byte| *byte == 0) => {
                return Err(ForecastJournalError::UnexpectedZeroVerifierDigest);
            }
            _ => {}
        }

        self.outcomes.insert(outcome.forecast_id, outcome);
        Ok(())
    }

    /// Iterate immutable forecast receipts in stable ForecastId order.
    pub fn iter_forecasts(&self) -> impl Iterator<Item = &ForecastReceipt> {
        self.forecasts.values()
    }

    pub fn forecast(&self, id: ForecastId) -> Option<&ForecastReceipt> {
        self.forecasts.get(&id)
    }

    pub fn outcome(&self, id: ForecastId) -> Option<&ForecastOutcomeReceipt> {
        self.outcomes.get(&id)
    }

    pub fn forecast_count(&self) -> usize {
        self.forecasts.len()
    }

    pub fn outcome_count(&self) -> usize {
        self.outcomes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CensorReason, ShiftStatus};

    fn forecast() -> RemainingCostForecast {
        RemainingCostForecast {
            pre_state_digest: [1; 32],
            frontier_digest: [2; 32],
            model_digest: [3; 32],
            calibration_digest: [4; 32],
            completion_probability: 0.95,
            remaining_q50: ResourceVector::new(20, 1, 10, 0, 30, 40),
            remaining_q90: ResourceVector::new(40, 2, 20, 1, 60, 80),
            remaining_q95: ResourceVector::new(50, 3, 25, 2, 75, 100),
            mandatory_verification_reserve: ResourceVector::new(5, 1, 2, 0, 5, 10),
            support_count: 128,
            shift_status: ShiftStatus::InDomain,
        }
    }

    fn receipt(id: u64, run_id: u64, sequence: u64) -> ForecastReceipt {
        ForecastReceipt {
            id: ForecastId(id),
            run_id,
            decision_sequence: sequence,
            task_family: 7,
            operation_family: 3,
            forecast: forecast(),
        }
    }

    fn completed_outcome(id: u64, sequence: u64) -> ForecastOutcomeReceipt {
        ForecastOutcomeReceipt {
            forecast_id: ForecastId(id),
            observed_sequence: sequence,
            observed_work: ResourceVector::new(18, 1, 9, 0, 25, 36),
            terminal: TerminalOutcome::SufficientProof,
            evidence_digest: [5; 32],
            meter_digest: [6; 32],
            verifier_digest: Some([7; 32]),
        }
    }

    #[test]
    fn forecast_and_outcome_are_separate_append_only_records() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 1)).unwrap();
        journal.append_outcome(completed_outcome(1, 2)).unwrap();

        let stored = journal.forecast(ForecastId(1)).unwrap();
        assert_eq!(stored.forecast.completion_probability, 0.95);
        assert_eq!(stored.decision_sequence, 1);
        assert_eq!(
            journal.outcome(ForecastId(1)).unwrap().terminal,
            TerminalOutcome::SufficientProof
        );
        assert_eq!(journal.forecast_count(), 1);
        assert_eq!(journal.outcome_count(), 1);
    }

    #[test]
    fn duplicate_forecast_identity_and_run_sequence_are_rejected() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 1)).unwrap();
        assert_eq!(
            journal.append_forecast(receipt(1, 10, 2)),
            Err(ForecastJournalError::DuplicateForecastId)
        );
        assert_eq!(
            journal.append_forecast(receipt(2, 10, 1)),
            Err(ForecastJournalError::DuplicateDecisionSequence)
        );
        assert_eq!(journal.forecast_count(), 1);
    }

    #[test]
    fn decision_sequences_must_increase_per_run() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 2)).unwrap();
        assert_eq!(
            journal.append_forecast(receipt(2, 10, 1)),
            Err(ForecastJournalError::NonIncreasingDecisionSequence)
        );
        assert_eq!(journal.forecast_count(), 1);
    }

    #[test]
    fn outcomes_cannot_precede_or_be_detached_from_forecasts() {
        let mut journal = ForecastJournal::default();
        assert_eq!(
            journal.append_outcome(completed_outcome(9, 2)),
            Err(ForecastJournalError::UnknownForecast)
        );
        journal.append_forecast(receipt(1, 10, 3)).unwrap();
        assert_eq!(
            journal.append_outcome(completed_outcome(1, 3)),
            Err(ForecastJournalError::OutcomePrecedesForecast)
        );
        assert_eq!(journal.outcome_count(), 0);
    }

    #[test]
    fn verified_completion_requires_independent_verifier_digest() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 1)).unwrap();
        let mut outcome = completed_outcome(1, 2);
        outcome.verifier_digest = None;
        assert_eq!(
            journal.append_outcome(outcome),
            Err(ForecastJournalError::MissingVerifierDigest)
        );
        assert_eq!(journal.outcome_count(), 0);
    }

    #[test]
    fn censored_outcome_keeps_lower_bound_without_fabricating_verification() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 1)).unwrap();
        let outcome = ForecastOutcomeReceipt {
            forecast_id: ForecastId(1),
            observed_sequence: 4,
            observed_work: ResourceVector::new(12, 2, 40, 0, 70, 0),
            terminal: TerminalOutcome::RightCensored(CensorReason::BudgetExhausted),
            evidence_digest: [8; 32],
            meter_digest: [9; 32],
            verifier_digest: None,
        };
        journal.append_outcome(outcome).unwrap();
        assert_eq!(
            journal.outcome(ForecastId(1)).unwrap().terminal,
            TerminalOutcome::RightCensored(CensorReason::BudgetExhausted)
        );
        assert_eq!(
            journal.outcome(ForecastId(1)).unwrap().observed_work.tokens,
            12
        );
    }

    #[test]
    fn outcome_is_single_assignment_and_cannot_rewrite_a_forecast() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 1)).unwrap();
        let original_probability = journal
            .forecast(ForecastId(1))
            .unwrap()
            .forecast
            .completion_probability;
        journal.append_outcome(completed_outcome(1, 2)).unwrap();
        assert_eq!(
            journal.append_outcome(completed_outcome(1, 3)),
            Err(ForecastJournalError::OutcomeAlreadyRecorded)
        );
        assert_eq!(
            journal
                .forecast(ForecastId(1))
                .unwrap()
                .forecast
                .completion_probability,
            original_probability
        );
    }

    #[test]
    fn missing_meter_or_evidence_digest_is_rejected() {
        let mut journal = ForecastJournal::default();
        journal.append_forecast(receipt(1, 10, 1)).unwrap();
        let mut outcome = completed_outcome(1, 2);
        outcome.meter_digest = [0; 32];
        assert_eq!(
            journal.append_outcome(outcome),
            Err(ForecastJournalError::MissingMeterDigest)
        );
        let mut outcome = completed_outcome(1, 2);
        outcome.evidence_digest = [0; 32];
        assert_eq!(
            journal.append_outcome(outcome),
            Err(ForecastJournalError::MissingEvidenceDigest)
        );
    }
}
