use std::collections::BTreeSet;

use crate::{
    ActionTrajectory, PrefixOutcomeLabel, ResourceVector, TrajectoryActionKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterventionAssignmentEvidence {
    DeterministicSimulation {
        simulator_digest: [u8; 32],
    },
    RandomizedLiveTrial {
        protocol_digest: [u8; 32],
        allocation_receipt_digest: [u8; 32],
    },
    ObservationalMatchedRuns {
        protocol_digest: [u8; 32],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterventionInterpretation {
    SimulatedContrast,
    RandomizedAssignmentEffect,
    ObservationalAssociationOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedResourceDelta {
    pub tokens: i128,
    pub tool_calls: i128,
    pub cpu_ms: i128,
    pub gpu_ms: i128,
    pub wall_ms: i128,
    pub micro_usd: i128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedInterventionContrast {
    pub baseline_operation_family: u32,
    pub alternative_operation_family: u32,
    /// Alternative minus baseline. Negative means lower measured cost for the
    /// alternative; positive means higher measured cost.
    pub resource_cost_delta: SignedResourceDelta,
    /// Positive means the alternative reduced more unresolved obligations
    /// immediately after the first operation than the baseline did.
    pub proof_progress_delta: i128,
    pub interpretation: InterventionInterpretation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchedInterventionSet {
    pub experiment_digest: [u8; 32],
    pub task_family: u32,
    pub seed: u64,
    pub initial_budget: Option<ResourceVector>,
    pub pre_state_digest: [u8; 32],
    pub frontier_digest: [u8; 32],
    pub frontier_features: crate::ProofFrontierFeatures,
    pub candidate_pool_digest: [u8; 32],
    pub admissible_operation_families: Vec<u32>,
    pub assignment: InterventionAssignmentEvidence,
    /// One isolated trajectory for each admissible operation family.
    pub trials: Vec<ActionTrajectory>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterventionSetError {
    MissingExperimentDigest,
    MissingAssignmentDigest,
    EmptyCandidatePool,
    CandidatePoolNotCanonical,
    TrialCountMismatch,
    DuplicateRunId,
    DuplicateSelectedOperation,
    MissingSelectedOperation,
    TaskFamilyMismatch,
    SeedMismatch,
    InitialBudgetMismatch,
    PreStateMismatch,
    FrontierMismatch,
    FrontierFeatureMismatch,
    CandidatePoolDigestMismatch,
    CandidatePoolMismatch,
    TrialDidNotStartWithProofOperation,
    TrialSelectedOperationNotInPool,
    TrialOutcomeNotVerified,
    ArithmeticOverflow,
}

impl std::fmt::Display for InterventionSetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InterventionSetError {}

impl MatchedInterventionSet {
    pub fn interpretation(&self) -> InterventionInterpretation {
        match self.assignment {
            InterventionAssignmentEvidence::DeterministicSimulation { .. } => {
                InterventionInterpretation::SimulatedContrast
            }
            InterventionAssignmentEvidence::RandomizedLiveTrial { .. } => {
                InterventionInterpretation::RandomizedAssignmentEffect
            }
            InterventionAssignmentEvidence::ObservationalMatchedRuns { .. } => {
                InterventionInterpretation::ObservationalAssociationOnly
            }
        }
    }

    pub fn validate(&self) -> Result<(), InterventionSetError> {
        if is_zero(self.experiment_digest) {
            return Err(InterventionSetError::MissingExperimentDigest);
        }
        let assignment_digest_valid = match self.assignment {
            InterventionAssignmentEvidence::DeterministicSimulation {
                simulator_digest,
            } => !is_zero(simulator_digest),
            InterventionAssignmentEvidence::RandomizedLiveTrial {
                protocol_digest,
                allocation_receipt_digest,
            } => !is_zero(protocol_digest) && !is_zero(allocation_receipt_digest),
            InterventionAssignmentEvidence::ObservationalMatchedRuns {
                protocol_digest,
            } => !is_zero(protocol_digest),
        };
        if !assignment_digest_valid {
            return Err(InterventionSetError::MissingAssignmentDigest);
        }
        if self.admissible_operation_families.is_empty() {
            return Err(InterventionSetError::EmptyCandidatePool);
        }
        if self
            .admissible_operation_families
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(InterventionSetError::CandidatePoolNotCanonical);
        }
        if self.trials.len() != self.admissible_operation_families.len() {
            return Err(InterventionSetError::TrialCountMismatch);
        }

        let mut run_ids = BTreeSet::new();
        let mut selected_families = BTreeSet::new();
        for trial in &self.trials {
            trial
                .validate()
                .map_err(|_| InterventionSetError::TrialDidNotStartWithProofOperation)?;
            if !run_ids.insert(trial.run_id) {
                return Err(InterventionSetError::DuplicateRunId);
            }
            if trial.task_family != self.task_family {
                return Err(InterventionSetError::TaskFamilyMismatch);
            }
            if trial.seed != self.seed {
                return Err(InterventionSetError::SeedMismatch);
            }
            if trial.initial_budget != self.initial_budget {
                return Err(InterventionSetError::InitialBudgetMismatch);
            }

            let first = trial
                .actions
                .first()
                .ok_or(InterventionSetError::TrialDidNotStartWithProofOperation)?;
            if first.kind != TrajectoryActionKind::ProofOperation {
                return Err(InterventionSetError::TrialDidNotStartWithProofOperation);
            }
            if first.pre_state_digest != self.pre_state_digest {
                return Err(InterventionSetError::PreStateMismatch);
            }
            if first.frontier_digest != self.frontier_digest {
                return Err(InterventionSetError::FrontierMismatch);
            }
            if first.frontier_features != self.frontier_features {
                return Err(InterventionSetError::FrontierFeatureMismatch);
            }
            if first.candidate_pool_digest != self.candidate_pool_digest {
                return Err(InterventionSetError::CandidatePoolDigestMismatch);
            }
            if first.admissible_operation_families != self.admissible_operation_families {
                return Err(InterventionSetError::CandidatePoolMismatch);
            }
            if !self
                .admissible_operation_families
                .contains(&first.operation_family)
            {
                return Err(InterventionSetError::TrialSelectedOperationNotInPool);
            }
            if !selected_families.insert(first.operation_family) {
                return Err(InterventionSetError::DuplicateSelectedOperation);
            }
        }

        for family in &self.admissible_operation_families {
            if !selected_families.contains(family) {
                return Err(InterventionSetError::MissingSelectedOperation);
            }
        }
        Ok(())
    }

    /// Compute resource-cost and immediate proof-frontier contrasts against
    /// the lowest-ID candidate as the deterministic baseline. This method
    /// requires verified outcomes for every trial; partial/censored sets must
    /// be analysed with a separate design rather than imputed.
    pub fn verified_contrasts(
        &self,
    ) -> Result<Vec<PairedInterventionContrast>, InterventionSetError> {
        self.validate()?;
        let mut observed = Vec::with_capacity(self.trials.len());
        for trial in &self.trials {
            let first = &trial.actions[0];
            let label = trial
                .outcome_label(0)
                .map_err(|_| InterventionSetError::TrialOutcomeNotVerified)?;
            let cost = match label {
                PrefixOutcomeLabel::VerifiedRemainingCost { cost, .. } => cost,
                _ => return Err(InterventionSetError::TrialOutcomeNotVerified),
            };
            let next_frontier = trial
                .actions
                .get(1)
                .ok_or(InterventionSetError::TrialOutcomeNotVerified)?
                .frontier_features;
            let progress = i128::try_from(first.frontier_features.unresolved_obligations)
                .map_err(|_| InterventionSetError::ArithmeticOverflow)?
                .checked_sub(
                    i128::try_from(next_frontier.unresolved_obligations)
                        .map_err(|_| InterventionSetError::ArithmeticOverflow)?,
                )
                .ok_or(InterventionSetError::ArithmeticOverflow)?;
            observed.push((
                first.operation_family,
                cost,
                progress,
            ));
        }
        observed.sort_by_key(|entry| entry.0);
        let (baseline_family, baseline_cost, baseline_progress) = observed[0];
        let interpretation = self.interpretation();
        let mut contrasts = Vec::with_capacity(observed.len().saturating_sub(1));
        for (family, cost, progress) in observed.into_iter().skip(1) {
            contrasts.push(PairedInterventionContrast {
                baseline_operation_family: baseline_family,
                alternative_operation_family: family,
                resource_cost_delta: resource_delta(cost, baseline_cost)?,
                proof_progress_delta: progress
                    .checked_sub(baseline_progress)
                    .ok_or(InterventionSetError::ArithmeticOverflow)?,
                interpretation,
            });
        }
        Ok(contrasts)
    }
}

fn resource_delta(
    alternative: ResourceVector,
    baseline: ResourceVector,
) -> Result<SignedResourceDelta, InterventionSetError> {
    let alt = resource_values(alternative);
    let base = resource_values(baseline);
    let mut delta = [0i128; 6];
    for index in 0..6 {
        delta[index] = i128::try_from(alt[index])
            .map_err(|_| InterventionSetError::ArithmeticOverflow)?
            .checked_sub(
                i128::try_from(base[index])
                    .map_err(|_| InterventionSetError::ArithmeticOverflow)?,
            )
            .ok_or(InterventionSetError::ArithmeticOverflow)?;
    }
    Ok(SignedResourceDelta {
        tokens: delta[0],
        tool_calls: delta[1],
        cpu_ms: delta[2],
        gpu_ms: delta[3],
        wall_ms: delta[4],
        micro_usd: delta[5],
    })
}

fn resource_values(value: ResourceVector) -> [u128; 6] {
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
    use crate::{
        ActionTrajectory, TrajectoryActionReceipt, TrajectoryTerminalRecord,
    };

    fn features(unresolved: u64) -> crate::ProofFrontierFeatures {
        crate::ProofFrontierFeatures {
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
        seq: u64,
        kind: TrajectoryActionKind,
        family: u32,
        pre: u8,
        post: u8,
        frontier: crate::ProofFrontierFeatures,
        candidates: Vec<u32>,
    ) -> TrajectoryActionReceipt {
        TrajectoryActionReceipt {
            decision_sequence: seq,
            kind,
            operation_id: format!("operation-{family}-{seq}"),
            operation_family: family,
            candidate_pool_digest: [3; 32],
            admissible_operation_families: candidates,
            pre_state_digest: [pre; 32],
            post_state_digest: [post; 32],
            frontier_digest: [4; 32],
            frontier_features: frontier,
            measured_cost: ResourceVector::new(
                if seq == 1 && family == 2 { 10 } else if seq == 1 { 15 } else { 5 },
                1,
                1,
                0,
                2,
                3,
            ),
            meter_receipt_digest: [5 + seq as u8; 32],
        }
    }

    fn trial(run: u64, family: u32, first_cost: u128) -> ActionTrajectory {
        let next_unresolved = if family == 2 { 1 } else { 0 };
        let first = action(
            1,
            TrajectoryActionKind::ProofOperation,
            family,
            1,
            2,
            features(2),
            vec![2, 3],
        );
        let mut first = first;
        first.measured_cost = ResourceVector::new(first_cost, 1, 1, 0, 2, 3);
        ActionTrajectory {
            run_id: run,
            task_family: 7,
            seed: 9,
            initial_budget: Some(ResourceVector::new(100, 10, 10, 10, 20, 30)),
            actions: vec![
                first,
                action(
                    2,
                    TrajectoryActionKind::ProofOperation,
                    8,
                    2,
                    3,
                    features(next_unresolved),
                    vec![8, 9],
                ),
                action(
                    3,
                    TrajectoryActionKind::IndependentVerifier,
                    90,
                    3,
                    4,
                    crate::ProofFrontierFeatures::default(),
                    vec![90],
                ),
            ],
            terminal: TrajectoryTerminalRecord {
                outcome: Some(crate::TerminalOutcome::SufficientProof),
                evaluator_receipt_digest: Some([6; 32]),
                completion_verifier_digest: Some([7; 32]),
            },
        }
    }

    fn set(assignment: InterventionAssignmentEvidence) -> MatchedInterventionSet {
        MatchedInterventionSet {
            experiment_digest: [1; 32],
            task_family: 7,
            seed: 9,
            initial_budget: Some(ResourceVector::new(100, 10, 10, 10, 20, 30)),
            pre_state_digest: [1; 32],
            frontier_digest: [4; 32],
            frontier_features: features(2),
            candidate_pool_digest: [3; 32],
            admissible_operation_families: vec![2, 3],
            assignment,
            trials: vec![trial(100, 2, 10), trial(101, 3, 15)],
        }
    }

    #[test]
    fn matched_set_requires_identical_pre_state_and_candidate_pool() {
        let matched = set(InterventionAssignmentEvidence::DeterministicSimulation {
            simulator_digest: [8; 32],
        });
        assert_eq!(matched.validate(), Ok(()));

        let mut changed = matched.clone();
        changed.trials[1].actions[0].pre_state_digest = [9; 32];
        assert_eq!(
            changed.validate(),
            Err(InterventionSetError::PreStateMismatch)
        );

        let mut wrong_pool = matched.clone();
        wrong_pool.trials[1].actions[0].admissible_operation_families = vec![2, 4];
        assert_eq!(
            wrong_pool.validate(),
            Err(InterventionSetError::CandidatePoolMismatch)
        );
    }

    #[test]
    fn verified_contrasts_are_labelled_as_simulated_not_causal() {
        let matched = set(InterventionAssignmentEvidence::DeterministicSimulation {
            simulator_digest: [8; 32],
        });
        let contrasts = matched.verified_contrasts().unwrap();
        assert_eq!(contrasts.len(), 1);
        assert_eq!(contrasts[0].baseline_operation_family, 2);
        assert_eq!(contrasts[0].alternative_operation_family, 3);
        assert_eq!(contrasts[0].resource_cost_delta.tokens, 5);
        assert_eq!(contrasts[0].proof_progress_delta, 1);
        assert_eq!(
            contrasts[0].interpretation,
            InterventionInterpretation::SimulatedContrast
        );
    }

    #[test]
    fn interpretation_follows_recorded_assignment_evidence() {
        assert_eq!(
            set(InterventionAssignmentEvidence::RandomizedLiveTrial {
                protocol_digest: [8; 32],
                allocation_receipt_digest: [9; 32],
            })
            .interpretation(),
            InterventionInterpretation::RandomizedAssignmentEffect
        );
        assert_eq!(
            set(InterventionAssignmentEvidence::ObservationalMatchedRuns {
                protocol_digest: [8; 32],
            })
            .interpretation(),
            InterventionInterpretation::ObservationalAssociationOnly
        );
    }

    #[test]
    fn incomplete_or_duplicated_alternative_coverage_is_rejected() {
        let mut incomplete =
            set(InterventionAssignmentEvidence::DeterministicSimulation {
                simulator_digest: [8; 32],
            });
        incomplete.trials.pop();
        assert_eq!(
            incomplete.validate(),
            Err(InterventionSetError::TrialCountMismatch)
        );

        let mut duplicate =
            set(InterventionAssignmentEvidence::DeterministicSimulation {
                simulator_digest: [8; 32],
            });
        duplicate.trials[1].actions[0].operation_family = 2;
        assert_eq!(
            duplicate.validate(),
            Err(InterventionSetError::DuplicateSelectedOperation)
        );
    }

    #[test]
    fn censored_outcomes_cannot_be_silently_used_as_effect_labels() {
        let mut matched =
            set(InterventionAssignmentEvidence::DeterministicSimulation {
                simulator_digest: [8; 32],
            });
        matched.trials[1].terminal.outcome =
            Some(crate::TerminalOutcome::RightCensored(crate::CensorReason::Timeout));
        matched.trials[1].terminal.completion_verifier_digest = None;
        assert_eq!(
            matched.verified_contrasts(),
            Err(InterventionSetError::TrialOutcomeNotVerified)
        );
    }
}
