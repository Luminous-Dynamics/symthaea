use crate::{
    HistoricalCostBaseline, ProofFrontierFeatures, ResourceVector, TerminalOutcome,
};

const RESOURCE_DIMENSIONS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryActionKind {
    ProofOperation,
    IndependentVerifier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryActionReceipt {
    pub decision_sequence: u64,
    pub kind: TrajectoryActionKind,
    pub operation_id: String,
    pub operation_family: u32,
    pub candidate_pool_digest: [u8; 32],
    pub admissible_operation_families: Vec<u32>,
    pub pre_state_digest: [u8; 32],
    pub post_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub frontier_features: ProofFrontierFeatures,
    pub measured_cost: ResourceVector,
    pub meter_receipt_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrajectoryTerminalRecord {
    pub outcome: Option<TerminalOutcome>,
    pub evaluator_receipt_digest: Option<[u8; 32]>,
    pub completion_verifier_digest: Option<[u8; 32]>,
}

/// Full trajectory held by the replay engine. Never passed to a forecaster.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionTrajectory {
    pub run_id: u64,
    pub task_family: u32,
    pub seed: u64,
    pub initial_budget: Option<ResourceVector>,
    pub actions: Vec<TrajectoryActionReceipt>,
    pub terminal: TrajectoryTerminalRecord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedActionSummary {
    pub decision_sequence: u64,
    pub operation_id: String,
    pub operation_family: u32,
    pub kind: TrajectoryActionKind,
    pub measured_cost: ResourceVector,
    pub post_state_digest: [u8; 32],
}

/// Only data available immediately before the next operation. It contains no
/// suffix actions, eventual terminal type, or future evaluator/verifier data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixView {
    pub run_id: u64,
    pub task_family: u32,
    pub seed: u64,
    pub completed_action_count: usize,
    pub cumulative_spend: ResourceVector,
    pub last_observed_action_cost: Option<ResourceVector>,
    pub observed_action_history: Vec<ObservedActionSummary>,
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub frontier_features: ProofFrontierFeatures,
    pub candidate_pool_digest: [u8; 32],
    pub admissible_operation_families: Vec<u32>,
    pub remaining_budget: Option<ResourceVector>,
    pub budget_exhausted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCostForecast {
    pub q50: ResourceVector,
    pub q90: ResourceVector,
    pub q95: ResourceVector,
}

impl PrefixCostForecast {
    pub fn validate(self) -> Result<(), PrefixForecasterError> {
        let q50 = resource_values(self.q50);
        let q90 = resource_values(self.q90);
        let q95 = resource_values(self.q95);
        for index in 0..RESOURCE_DIMENSIONS {
            if q50[index] > q90[index] || q90[index] > q95[index] {
                return Err(PrefixForecasterError::NonMonotoneQuantiles);
            }
        }
        Ok(())
    }
}

pub trait PrefixForecaster {
    fn model_digest(&self) -> [u8; 32];

    fn predict(
        &self,
        prefix: &PrefixView,
    ) -> Result<PrefixCostForecast, PrefixForecasterError>;
}

/// Simple baseline using the last observed action cost, or a declared cold
/// start cost, multiplied by unresolved obligations plus verifier work.
/// This is a heuristic, not a calibrated predictor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastObservedActionForecaster {
    pub model_digest: [u8; 32],
    pub cold_start_action_cost: ResourceVector,
    pub q90_multiplier: u64,
    pub q95_multiplier: u64,
}

impl LastObservedActionForecaster {
    pub fn new(
        model_digest: [u8; 32],
        cold_start_action_cost: ResourceVector,
    ) -> Result<Self, PrefixForecasterError> {
        let result = Self {
            model_digest,
            cold_start_action_cost,
            q90_multiplier: 2,
            q95_multiplier: 3,
        };
        result.validate()?;
        Ok(result)
    }

    fn validate(self) -> Result<(), PrefixForecasterError> {
        if is_zero(self.model_digest) {
            return Err(PrefixForecasterError::MissingModelDigest);
        }
        if self.q90_multiplier == 0 || self.q95_multiplier < self.q90_multiplier {
            return Err(PrefixForecasterError::InvalidHeuristicMultipliers);
        }
        Ok(())
    }
}

impl PrefixForecaster for LastObservedActionForecaster {
    fn model_digest(&self) -> [u8; 32] {
        self.model_digest
    }

    fn predict(
        &self,
        prefix: &PrefixView,
    ) -> Result<PrefixCostForecast, PrefixForecasterError> {
        self.validate()?;
        let unit_cost = prefix
            .last_observed_action_cost
            .unwrap_or(self.cold_start_action_cost);
        let work_units = prefix
            .frontier_features
            .unresolved_obligations
            .checked_add(prefix.frontier_features.mandatory_verifier_operations)
            .ok_or(PrefixForecasterError::ArithmeticOverflow)?
            .max(1);
        let q50 = checked_scale(unit_cost, work_units)
            .ok_or(PrefixForecasterError::ArithmeticOverflow)?;
        let q90 = checked_scale(q50, u128::from(self.q90_multiplier))
            .ok_or(PrefixForecasterError::ArithmeticOverflow)?;
        let q95 = checked_scale(q50, u128::from(self.q95_multiplier))
            .ok_or(PrefixForecasterError::ArithmeticOverflow)?;
        let result = PrefixCostForecast { q50, q90, q95 };
        result.validate()?;
        Ok(result)
    }
}

impl PrefixForecaster for HistoricalCostBaseline {
    fn model_digest(&self) -> [u8; 32] {
        self.model_digest
    }

    fn predict(
        &self,
        prefix: &PrefixView,
    ) -> Result<PrefixCostForecast, PrefixForecasterError> {
        let profile = HistoricalCostBaseline::predict(self, prefix.task_family).profile;
        let result = PrefixCostForecast {
            q50: profile.q50,
            q90: profile.q90,
            q95: profile.q95,
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixOutcomeLabel {
    VerifiedRemainingCost {
        cost: ResourceVector,
        verifier_digest: [u8; 32],
    },
    RightCensoredLowerBound {
        observed_work: ResourceVector,
        reason: crate::CensorReason,
    },
    CompetingTerminal {
        observed_work: ResourceVector,
        reason: crate::CompetingReason,
    },
    Pending {
        observed_work: ResourceVector,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixReplayRow {
    pub prefix: PrefixView,
    pub model_digest: [u8; 32],
    pub forecast: PrefixCostForecast,
    pub label: PrefixOutcomeLabel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryReplayResult {
    pub run_id: u64,
    pub task_family: u32,
    pub rows: Vec<PrefixReplayRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryReplayError {
    InvalidRunId,
    EmptyTrajectory,
    InvalidActionSequence,
    EmptyOperationId,
    MissingCandidatePoolDigest,
    EmptyCandidatePool,
    CandidatePoolNotCanonical,
    SelectedOperationNotAdmissible,
    MissingPreStateDigest,
    MissingPostStateDigest,
    BrokenStateChain,
    MissingFrontierDigest,
    InvalidFrontierFeatures,
    MissingMeterReceipt,
    MissingTerminalEvidence,
    MissingCompletionVerifier,
    UnexpectedCompletionVerifier,
    MissingVerifierAction,
    VerifierActionNotTerminal,
    ArithmeticOverflow,
    PrefixIndexOutOfRange,
}

impl std::fmt::Display for TrajectoryReplayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for TrajectoryReplayError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixForecasterError {
    MissingModelDigest,
    InvalidHeuristicMultipliers,
    ArithmeticOverflow,
    NonMonotoneQuantiles,
}

impl std::fmt::Display for PrefixForecasterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PrefixForecasterError {}

impl ActionTrajectory {
    pub fn validate(&self) -> Result<(), TrajectoryReplayError> {
        if self.run_id == 0 {
            return Err(TrajectoryReplayError::InvalidRunId);
        }
        if self.actions.is_empty() {
            return Err(TrajectoryReplayError::EmptyTrajectory);
        }

        for (index, action) in self.actions.iter().enumerate() {
            let expected = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or(TrajectoryReplayError::ArithmeticOverflow)?;
            if action.decision_sequence != expected {
                return Err(TrajectoryReplayError::InvalidActionSequence);
            }
            if action.operation_id.trim().is_empty() {
                return Err(TrajectoryReplayError::EmptyOperationId);
            }
            if is_zero(action.candidate_pool_digest) {
                return Err(TrajectoryReplayError::MissingCandidatePoolDigest);
            }
            if action.admissible_operation_families.is_empty() {
                return Err(TrajectoryReplayError::EmptyCandidatePool);
            }
            if action
                .admissible_operation_families
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(TrajectoryReplayError::CandidatePoolNotCanonical);
            }
            if !action
                .admissible_operation_families
                .contains(&action.operation_family)
            {
                return Err(TrajectoryReplayError::SelectedOperationNotAdmissible);
            }
            if is_zero(action.pre_state_digest) {
                return Err(TrajectoryReplayError::MissingPreStateDigest);
            }
            if is_zero(action.post_state_digest) {
                return Err(TrajectoryReplayError::MissingPostStateDigest);
            }
            if is_zero(action.frontier_digest) {
                return Err(TrajectoryReplayError::MissingFrontierDigest);
            }
            action
                .frontier_features
                .validate()
                .map_err(|_| TrajectoryReplayError::InvalidFrontierFeatures)?;
            if is_zero(action.meter_receipt_digest) {
                return Err(TrajectoryReplayError::MissingMeterReceipt);
            }
            if index > 0
                && self.actions[index - 1].post_state_digest != action.pre_state_digest
            {
                return Err(TrajectoryReplayError::BrokenStateChain);
            }
            if action.kind == TrajectoryActionKind::IndependentVerifier
                && index + 1 != self.actions.len()
            {
                return Err(TrajectoryReplayError::VerifierActionNotTerminal);
            }
        }

        match self.terminal.outcome {
            Some(TerminalOutcome::SufficientProof) => {
                if self
                    .terminal
                    .evaluator_receipt_digest
                    .is_none_or(is_zero)
                {
                    return Err(TrajectoryReplayError::MissingTerminalEvidence);
                }
                if self
                    .terminal
                    .completion_verifier_digest
                    .is_none_or(is_zero)
                {
                    return Err(TrajectoryReplayError::MissingCompletionVerifier);
                }
                if self.actions.last().map(|action| action.kind)
                    != Some(TrajectoryActionKind::IndependentVerifier)
                {
                    return Err(TrajectoryReplayError::MissingVerifierAction);
                }
            }
            Some(_) => {
                if self
                    .terminal
                    .evaluator_receipt_digest
                    .is_none_or(is_zero)
                {
                    return Err(TrajectoryReplayError::MissingTerminalEvidence);
                }
                if self.terminal.completion_verifier_digest.is_some() {
                    return Err(TrajectoryReplayError::UnexpectedCompletionVerifier);
                }
                if self
                    .actions
                    .iter()
                    .any(|action| action.kind == TrajectoryActionKind::IndependentVerifier)
                {
                    return Err(TrajectoryReplayError::VerifierActionNotTerminal);
                }
            }
            None => {
                if self.terminal.evaluator_receipt_digest.is_some()
                    || self.terminal.completion_verifier_digest.is_some()
                {
                    return Err(TrajectoryReplayError::MissingTerminalEvidence);
                }
                if self
                    .actions
                    .iter()
                    .any(|action| action.kind == TrajectoryActionKind::IndependentVerifier)
                {
                    return Err(TrajectoryReplayError::VerifierActionNotTerminal);
                }
            }
        }
        Ok(())
    }

    /// Construct the view immediately before the action at next_action_index.
    /// No action at or after that index, nor any terminal label, is copied.
    pub fn prefix_view(
        &self,
        next_action_index: usize,
    ) -> Result<PrefixView, TrajectoryReplayError> {
        self.validate()?;
        let current = self
            .actions
            .get(next_action_index)
            .ok_or(TrajectoryReplayError::PrefixIndexOutOfRange)?;

        let mut spend = ResourceVector::ZERO;
        let mut history = Vec::with_capacity(next_action_index);
        for action in &self.actions[..next_action_index] {
            spend = spend
                .checked_add(action.measured_cost)
                .ok_or(TrajectoryReplayError::ArithmeticOverflow)?;
            history.push(ObservedActionSummary {
                decision_sequence: action.decision_sequence,
                operation_id: action.operation_id.clone(),
                operation_family: action.operation_family,
                kind: action.kind,
                measured_cost: action.measured_cost,
                post_state_digest: action.post_state_digest,
            });
        }
        let remaining_budget = self
            .initial_budget
            .and_then(|budget| budget.checked_sub(spend));
        let budget_exhausted = self.initial_budget.is_some() && remaining_budget.is_none();

        Ok(PrefixView {
            run_id: self.run_id,
            task_family: self.task_family,
            seed: self.seed,
            completed_action_count: next_action_index,
            cumulative_spend: spend,
            last_observed_action_cost: history.last().map(|item| item.measured_cost),
            observed_action_history: history,
            pre_state_digest: current.pre_state_digest,
            frontier_digest: current.frontier_digest,
            frontier_features: current.frontier_features,
            candidate_pool_digest: current.candidate_pool_digest,
            admissible_operation_families: current.admissible_operation_families.clone(),
            remaining_budget,
            budget_exhausted,
        })
    }

    /// Called only after forecasting. This derives a label from the sealed
    /// suffix, outside the forecaster-facing interface.
    pub fn outcome_label(
        &self,
        next_action_index: usize,
    ) -> Result<PrefixOutcomeLabel, TrajectoryReplayError> {
        self.validate()?;
        if next_action_index >= self.actions.len() {
            return Err(TrajectoryReplayError::PrefixIndexOutOfRange);
        }
        let mut observed = ResourceVector::ZERO;
        for action in &self.actions[next_action_index..] {
            observed = observed
                .checked_add(action.measured_cost)
                .ok_or(TrajectoryReplayError::ArithmeticOverflow)?;
        }
        match self.terminal.outcome {
            Some(TerminalOutcome::SufficientProof) => {
                Ok(PrefixOutcomeLabel::VerifiedRemainingCost {
                    cost: observed,
                    verifier_digest: self
                        .terminal
                        .completion_verifier_digest
                        .ok_or(TrajectoryReplayError::MissingCompletionVerifier)?,
                })
            }
            Some(TerminalOutcome::RightCensored(reason)) => {
                Ok(PrefixOutcomeLabel::RightCensoredLowerBound {
                    observed_work: observed,
                    reason,
                })
            }
            Some(TerminalOutcome::CompetingTerminal(reason)) => {
                Ok(PrefixOutcomeLabel::CompetingTerminal {
                    observed_work: observed,
                    reason,
                })
            }
            None => Ok(PrefixOutcomeLabel::Pending {
                observed_work: observed,
            }),
        }
    }
}

pub struct PrefixReplayEngine;

impl PrefixReplayEngine {
    /// Calls the forecaster using prefix data only, then separately attaches
    /// the sealed suffix-derived label for offline scoring.
    pub fn replay<F: PrefixForecaster>(
        trajectory: &ActionTrajectory,
        forecaster: &F,
    ) -> Result<TrajectoryReplayResult, PrefixReplayEngineError> {
        trajectory
            .validate()
            .map_err(PrefixReplayEngineError::Trajectory)?;
        let model_digest = forecaster.model_digest();
        if is_zero(model_digest) {
            return Err(PrefixReplayEngineError::Forecaster(
                PrefixForecasterError::MissingModelDigest,
            ));
        }
        let mut rows = Vec::with_capacity(trajectory.actions.len());
        for index in 0..trajectory.actions.len() {
            let prefix = trajectory
                .prefix_view(index)
                .map_err(PrefixReplayEngineError::Trajectory)?;
            let forecast = forecaster
                .predict(&prefix)
                .map_err(PrefixReplayEngineError::Forecaster)?;
            forecast
                .validate()
                .map_err(PrefixReplayEngineError::Forecaster)?;
            let label = trajectory
                .outcome_label(index)
                .map_err(PrefixReplayEngineError::Trajectory)?;
            rows.push(PrefixReplayRow {
                prefix,
                model_digest,
                forecast,
                label,
            });
        }
        Ok(TrajectoryReplayResult {
            run_id: trajectory.run_id,
            task_family: trajectory.task_family,
            rows,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixReplayEngineError {
    Trajectory(TrajectoryReplayError),
    Forecaster(PrefixForecasterError),
}

impl std::fmt::Display for PrefixReplayEngineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PrefixReplayEngineError {}

fn checked_scale(value: ResourceVector, multiplier: u128) -> Option<ResourceVector> {
    Some(ResourceVector::new(
        value.tokens.checked_mul(multiplier)?,
        value.tool_calls.checked_mul(multiplier)?,
        value.cpu_ms.checked_mul(multiplier)?,
        value.gpu_ms.checked_mul(multiplier)?,
        value.wall_ms.checked_mul(multiplier)?,
        value.micro_usd.checked_mul(multiplier)?,
    ))
}

fn resource_values(value: ResourceVector) -> [u128; RESOURCE_DIMENSIONS] {
    [
        value.tokens,
        value.tool_calls,
        value.cpu_ms,
        value.gpu_ms,
        value.wall_ms,
        value.micro_usd,
    ]
}

fn is_zero(digest: [u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CensorReason, CompetingReason};

    fn features(unresolved: u64) -> ProofFrontierFeatures {
        ProofFrontierFeatures {
            unresolved_obligations: unresolved,
            mandatory_obligations: unresolved,
            maximum_dependency_depth: unresolved,
            dependency_edges: unresolved.saturating_sub(1),
            admissible_alternatives: 2,
            missing_evidence_requirements: 0,
            distinct_operation_families: 2,
            distinct_evidence_families: 1,
            mandatory_verifier_operations: 0,
        }
    }

    fn action(
        sequence: u64,
        kind: TrajectoryActionKind,
        family: u32,
        pre: u8,
        post: u8,
        cost: ResourceVector,
        frontier: ProofFrontierFeatures,
        candidates: Vec<u32>,
    ) -> TrajectoryActionReceipt {
        TrajectoryActionReceipt {
            decision_sequence: sequence,
            kind,
            operation_id: format!("operation-{sequence}"),
            operation_family: family,
            candidate_pool_digest: [30 + sequence as u8; 32],
            admissible_operation_families: candidates,
            pre_state_digest: [pre; 32],
            post_state_digest: [post; 32],
            frontier_digest: [40 + sequence as u8; 32],
            frontier_features: frontier,
            measured_cost: cost,
            meter_receipt_digest: [50 + sequence as u8; 32],
        }
    }

    fn trace(outcome: Option<TerminalOutcome>) -> ActionTrajectory {
        let success = outcome == Some(TerminalOutcome::SufficientProof);
        let mut actions = vec![
            action(
                1,
                TrajectoryActionKind::ProofOperation,
                2,
                1,
                2,
                ResourceVector::new(10, 1, 2, 0, 3, 4),
                features(2),
                vec![2, 3],
            ),
            action(
                2,
                TrajectoryActionKind::ProofOperation,
                3,
                2,
                3,
                ResourceVector::new(20, 1, 4, 1, 5, 6),
                features(1),
                vec![3, 4],
            ),
        ];
        if success {
            actions.push(action(
                3,
                TrajectoryActionKind::IndependentVerifier,
                90,
                3,
                4,
                ResourceVector::new(5, 1, 1, 0, 2, 3),
                ProofFrontierFeatures::default(),
                vec![90],
            ));
        }
        ActionTrajectory {
            run_id: 100,
            task_family: 7,
            seed: 42,
            initial_budget: Some(ResourceVector::new(100, 10, 20, 10, 40, 50)),
            actions,
            terminal: TrajectoryTerminalRecord {
                outcome,
                evaluator_receipt_digest: outcome.map(|_| [60; 32]),
                completion_verifier_digest: success.then_some([61; 32]),
            },
        }
    }

    fn forecaster() -> LastObservedActionForecaster {
        LastObservedActionForecaster::new(
            [70; 32],
            ResourceVector::new(2, 1, 1, 0, 1, 1),
        )
        .unwrap()
    }

    #[test]
    fn prefix_contains_only_observed_history_and_current_frontier() {
        let trajectory = trace(Some(TerminalOutcome::SufficientProof));
        let prefix = trajectory.prefix_view(1).unwrap();
        assert_eq!(prefix.completed_action_count, 1);
        assert_eq!(prefix.observed_action_history.len(), 1);
        assert_eq!(prefix.cumulative_spend.tokens, 10);
        assert_eq!(prefix.last_observed_action_cost.unwrap().tokens, 10);
        assert_eq!(prefix.pre_state_digest, [2; 32]);
        assert_eq!(prefix.admissible_operation_families, vec![3, 4]);
        assert_eq!(prefix.remaining_budget.unwrap().tokens, 90);
    }

    #[test]
    fn changing_suffix_cost_does_not_change_earlier_prefix_or_forecast() {
        let original = trace(Some(TerminalOutcome::SufficientProof));
        let mut altered = original.clone();
        altered.actions[1].measured_cost =
            ResourceVector::new(200, 7, 9, 4, 12, 18);
        let original_prefix = original.prefix_view(1).unwrap();
        let altered_prefix = altered.prefix_view(1).unwrap();
        assert_eq!(original_prefix, altered_prefix);
        let predictor = forecaster();
        assert_eq!(
            predictor.predict(&original_prefix).unwrap(),
            predictor.predict(&altered_prefix).unwrap()
        );
        assert_ne!(
            original.outcome_label(1).unwrap(),
            altered.outcome_label(1).unwrap()
        );
    }

    #[test]
    fn terminal_type_is_not_visible_to_forecaster_prefix() {
        let complete = trace(Some(TerminalOutcome::SufficientProof));
        let censored = trace(Some(TerminalOutcome::RightCensored(CensorReason::Timeout)));
        assert_eq!(
            complete.prefix_view(0).unwrap(),
            censored.prefix_view(0).unwrap()
        );
        assert_ne!(
            complete.outcome_label(0).unwrap(),
            censored.outcome_label(0).unwrap()
        );
    }

    #[test]
    fn verified_remaining_cost_includes_the_independent_verifier_action() {
        let trajectory = trace(Some(TerminalOutcome::SufficientProof));
        assert_eq!(
            trajectory.outcome_label(0).unwrap(),
            PrefixOutcomeLabel::VerifiedRemainingCost {
                cost: ResourceVector::new(35, 3, 7, 1, 10, 13),
                verifier_digest: [61; 32],
            }
        );
    }

    #[test]
    fn censored_competing_and_pending_outcomes_stay_distinct() {
        let censored = trace(Some(TerminalOutcome::RightCensored(CensorReason::Timeout)));
        assert!(matches!(
            censored.outcome_label(0).unwrap(),
            PrefixOutcomeLabel::RightCensoredLowerBound { .. }
        ));
        let competing = trace(Some(TerminalOutcome::CompetingTerminal(
            CompetingReason::PermissionRevoked,
        )));
        assert!(matches!(
            competing.outcome_label(0).unwrap(),
            PrefixOutcomeLabel::CompetingTerminal { .. }
        ));
        let pending = trace(None);
        assert!(matches!(
            pending.outcome_label(0).unwrap(),
            PrefixOutcomeLabel::Pending { .. }
        ));
    }

    #[test]
    fn replay_engine_returns_prefix_rows_and_separately_joined_labels() {
        let trajectory = trace(Some(TerminalOutcome::SufficientProof));
        let replay = PrefixReplayEngine::replay(&trajectory, &forecaster()).unwrap();
        assert_eq!(replay.rows.len(), 3);
        assert_eq!(replay.rows[0].label, trajectory.outcome_label(0).unwrap());
        assert_eq!(replay.rows[2].label, trajectory.outcome_label(2).unwrap());
        assert_eq!(replay.rows[0].model_digest, [70; 32]);
    }

    #[test]
    fn broken_state_chains_and_noncanonical_candidate_pools_are_rejected() {
        let mut broken = trace(Some(TerminalOutcome::SufficientProof));
        broken.actions[1].pre_state_digest = [99; 32];
        assert_eq!(
            broken.validate(),
            Err(TrajectoryReplayError::BrokenStateChain)
        );
        let mut noncanonical = trace(Some(TerminalOutcome::SufficientProof));
        noncanonical.actions[0].admissible_operation_families = vec![3, 2];
        assert_eq!(
            noncanonical.validate(),
            Err(TrajectoryReplayError::CandidatePoolNotCanonical)
        );
    }

    #[test]
    fn sufficient_proof_requires_final_verifier_action_and_digest() {
        let mut no_receipt = trace(Some(TerminalOutcome::SufficientProof));
        no_receipt.terminal.completion_verifier_digest = None;
        assert_eq!(
            no_receipt.validate(),
            Err(TrajectoryReplayError::MissingCompletionVerifier)
        );
        let mut no_action = trace(Some(TerminalOutcome::SufficientProof));
        no_action.actions.pop();
        assert_eq!(
            no_action.validate(),
            Err(TrajectoryReplayError::MissingVerifierAction)
        );
    }

    #[test]
    fn last_action_heuristic_uses_prior_cost_and_checked_scaling() {
        let predictor = forecaster();
        let prefix = trace(Some(TerminalOutcome::SufficientProof))
            .prefix_view(1)
            .unwrap();
        let estimate = predictor.predict(&prefix).unwrap();
        assert_eq!(estimate.q50.tokens, 20);
        assert_eq!(estimate.q90.tokens, 40);
        assert_eq!(estimate.q95.tokens, 60);

        let cold_start = trace(Some(TerminalOutcome::SufficientProof))
            .prefix_view(0)
            .unwrap();
        assert_eq!(
            predictor.predict(&cold_start).unwrap().q50.tokens,
            4
        );
    }

    #[test]
    fn budget_exhaustion_is_explicit() {
        let mut trajectory = trace(Some(TerminalOutcome::SufficientProof));
        trajectory.initial_budget = Some(ResourceVector::new(5, 1, 1, 1, 1, 1));
        let prefix = trajectory.prefix_view(1).unwrap();
        assert!(prefix.budget_exhausted);
        assert_eq!(prefix.remaining_budget, None);
    }

    #[test]
    fn arithmetic_overflow_in_labels_is_rejected() {
        let mut trajectory = trace(Some(TerminalOutcome::SufficientProof));
        trajectory.actions[0].measured_cost.tokens = u128::MAX;
        assert_eq!(
            trajectory.outcome_label(0),
            Err(TrajectoryReplayError::ArithmeticOverflow)
        );
    }
}
