//! Evaluator-only exact joint-action subset qualification.
//!
//! Candidate plans are scored against evaluator-measured outcomes and costs.
//! The candidate never receives this module, the hidden future stream, or the
//! evaluator's exact subset results. Exhaustive enumeration is limited to
//! small action sets and is an oracle, never a deployable candidate policy.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use sha2::{Digest, Sha256};

pub const MAX_EXACT_JOINT_ACTIONS: usize = 16;

/// Evaluator-owned action compatibility registry. Actions sharing a nonzero
/// group are mutually exclusive; group zero means independent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointActionDefinition {
    pub action_id: u64,
    pub conflict_group: u64,
    /// Digest of canonical evaluator-owned operation kind and parameters.
    pub action_semantics_sha256: [u8; 32],
}

/// Canonical evaluator-side digest of sorted action IDs, conflict groups and action-contract digests.
pub fn action_registry_sha256(
    registry: &[JointActionDefinition],
) -> Result<[u8; 32], JointBudgetError> {
    if registry.len() > MAX_EXACT_JOINT_ACTIONS {
        return Err(JointBudgetError::TooManyActions {
            requested: registry.len(),
            maximum: MAX_EXACT_JOINT_ACTIONS,
        });
    }
    let mut sorted = registry.to_vec();
    sorted.sort_unstable_by_key(|item| item.action_id);
    if sorted.iter().any(|item| item.action_id == 0) {
        return Err(JointBudgetError::ZeroActionId);
    }
    if let Some(pair) = sorted.windows(2).find(|pair| pair[0].action_id == pair[1].action_id) {
        return Err(JointBudgetError::DuplicateActionId {
            action_id: pair[0].action_id,
        });
    }
    let mut hasher = Sha256::new();
    hasher.update(b"SYMTHAEA-JOINT-ACTION-REGISTRY-V1\0");
    hasher.update(u32::try_from(sorted.len())
        .map_err(|_| JointBudgetError::ArithmeticOverflow)?
        .to_le_bytes());
    for item in sorted {
        if item.action_semantics_sha256 == [0; 32] {
            return Err(JointBudgetError::InvalidActionSemanticsDigest {
                action_id: item.action_id,
            });
        }
        hasher.update(item.action_id.to_le_bytes());
        hasher.update(item.conflict_group.to_le_bytes());
        hasher.update(item.action_semantics_sha256);
    }
    Ok(hasher.finalize().into())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JointResourceUse {
    pub persistent_bytes: u64,
    pub retained_byte_us: u128,
    pub io_bytes: u64,
    pub latency_us: u64,
    pub compute_units: u64,
    pub observations: u64,
    pub energy_nj: u64,
    pub verification_units: u64,
}

impl JointResourceUse {
    pub fn fits_within(self, budget: Self) -> bool {
        self.persistent_bytes <= budget.persistent_bytes
            && self.retained_byte_us <= budget.retained_byte_us
            && self.io_bytes <= budget.io_bytes
            && self.latency_us <= budget.latency_us
            && self.compute_units <= budget.compute_units
            && self.observations <= budget.observations
            && self.energy_nj <= budget.energy_nj
            && self.verification_units <= budget.verification_units
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JointExposure {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

impl JointExposure {
    pub(crate) fn validate(self, total_exposure_us: u64) -> Result<(), JointBudgetError> {
        let total = self.wrong_belief_us
            .checked_add(self.abstention_us)
            .and_then(|value| value.checked_add(self.missing_output_us))
            .ok_or(JointBudgetError::ArithmeticOverflow)?;
        if total > total_exposure_us {
            return Err(JointBudgetError::ExposureExceedsEpisode {
                measured_us: total,
                total_us: total_exposure_us,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointBudgetContext {
    pub episode_id: u64,
    pub task_sha256: [u8; 32],
    pub future_stream_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub initial_state_sha256: [u8; 32],
    pub candidate_feature_sha256: [u8; 32],
    /// Digest of the evaluator-owned action IDs and compatibility constraints.
    pub action_registry_sha256: [u8; 32],
    pub total_exposure_us: u64,
    pub resource_budget: JointResourceUse,
}

impl JointBudgetContext {
    pub(crate) fn validate(&self) -> Result<(), JointBudgetError> {
        if self.episode_id == 0
            || self.task_sha256 == [0; 32]
            || self.future_stream_sha256 == [0; 32]
            || self.model_sha256 == [0; 32]
            || self.initial_state_sha256 == [0; 32]
            || self.candidate_feature_sha256 == [0; 32]
            || self.action_registry_sha256 == [0; 32]
            || self.total_exposure_us == 0
        {
            return Err(JointBudgetError::InvalidContextCommitment);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointSubsetRun {
    pub context: JointBudgetContext,
    pub run_id: u64,
    /// Sorted, unique action IDs selected in this exact joint intervention.
    pub selected_action_ids: Vec<u64>,
    /// Evaluator-measured costs; not candidate estimates.
    pub measured_resources: JointResourceUse,
    pub outcome: JointExposure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointSubsetSummary {
    pub subset_mask: u64,
    pub selected_action_ids: Vec<u64>,
    pub measured_resources: JointResourceUse,
    pub outcome: JointExposure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointExposureRegret {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointBudgetQualification {
    /// Feasible nondominated outcomes and measured resource vectors.
    pub pareto_frontier: Vec<JointSubsetSummary>,
    /// Componentwise optima are deliberately separate; they may be different subsets.
    pub best_wrong_belief: JointSubsetSummary,
    pub best_abstention: JointSubsetSummary,
    pub best_missing_output: JointSubsetSummary,
    pub candidate_regret: JointExposureRegret,
    pub evaluated_subset_count: usize,
    pub feasible_subset_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JointBudgetError {
    TooManyActions { requested: usize, maximum: usize },
    ZeroActionId,
    InvalidActionSemanticsDigest { action_id: u64 },
    DuplicateActionId { action_id: u64 },
    UnsortedOrDuplicateSelection,
    UnknownActionId { action_id: u64 },
    ContextMismatch,
    DuplicateRunId { run_id: u64 },
    InvalidContextCommitment,
    IncompleteEnumeration { expected: usize, actual: usize },
    DuplicateSubset { mask: u64 },
    MissingSubset { mask: u64 },
    CandidateSubsetMissing,
    CandidateSubsetInfeasible,
    NoFeasibleSubset,
    ExposureExceedsEpisode { measured_us: u64, total_us: u64 },
    ArithmeticOverflow,
}

impl fmt::Display for JointBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyActions { requested, maximum } => write!(f, "exact joint oracle supports at most {maximum} actions, received {requested}"),
            Self::ZeroActionId => write!(f, "joint action ID must be nonzero"),
            Self::InvalidActionSemanticsDigest { action_id } => write!(f, "joint action {action_id} has a zero semantic contract digest"),
            Self::DuplicateActionId { action_id } => write!(f, "duplicate joint action ID {action_id}"),
            Self::UnsortedOrDuplicateSelection => write!(f, "joint action IDs must be strictly ascending and unique"),
            Self::UnknownActionId { action_id } => write!(f, "unknown joint action ID {action_id}"),
            Self::ContextMismatch => write!(f, "joint subset runs differ in task, future stream, model, initial state, feature snapshot, action registry, horizon or resource budget"),
            Self::DuplicateRunId { run_id } => write!(f, "duplicate joint subset run ID {run_id}"),
            Self::InvalidContextCommitment => write!(f, "joint oracle context contains a zero/missing identity or commitment"),
            Self::IncompleteEnumeration { expected, actual } => write!(f, "exact joint oracle requires {expected} subsets, received {actual}"),
            Self::DuplicateSubset { mask } => write!(f, "duplicate joint subset mask {mask:#x}"),
            Self::MissingSubset { mask } => write!(f, "missing joint subset mask {mask:#x}"),
            Self::CandidateSubsetMissing => write!(f, "candidate-selected joint action subset is absent from evaluator runs"),
            Self::CandidateSubsetInfeasible => write!(f, "candidate-selected joint action subset exceeds at least one hard resource ceiling"),
            Self::NoFeasibleSubset => write!(f, "joint oracle has no resource-feasible subset"),
            Self::ExposureExceedsEpisode { measured_us, total_us } => write!(f, "joint temporal exposure {measured_us} us exceeds episode horizon {total_us} us"),
            Self::ArithmeticOverflow => write!(f, "joint budget arithmetic overflowed"),
        }
    }
}

impl std::error::Error for JointBudgetError {}

/// Qualify a candidate subset against exhaustive evaluator-measured outcomes.
///
/// All 2^n combinations must be present for at most 16 atomic action IDs.
/// Costs and outcomes are read from evaluator-owned records. Infeasible oracle
/// subsets remain represented in the enumeration but are excluded from the
/// frontier and componentwise optimum. The candidate's own selected subset
/// must be feasible or qualification returns an explicit error.
pub fn qualify_joint_subsets(
    action_registry: &[JointActionDefinition],
    runs: &[JointSubsetRun],
    candidate_selected_action_ids: &[u64],
) -> Result<JointBudgetQualification, JointBudgetError> {
    if action_registry.len() > MAX_EXACT_JOINT_ACTIONS {
        return Err(JointBudgetError::TooManyActions {
            requested: action_registry.len(),
            maximum: MAX_EXACT_JOINT_ACTIONS,
        });
    }
    let mut sorted_registry = action_registry.to_vec();
    sorted_registry.sort_unstable_by_key(|definition| definition.action_id);
    let sorted_action_ids: Vec<u64> = sorted_registry.iter().map(|definition| definition.action_id).collect();
    if sorted_action_ids.contains(&0) {
        return Err(JointBudgetError::ZeroActionId);
    }
    if sorted_action_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(JointBudgetError::DuplicateActionId {
            action_id: sorted_action_ids.windows(2).find(|pair| pair[0] == pair[1]).unwrap()[0],
        });
    }

    let expected = 1_usize
        .checked_shl(u32::try_from(sorted_action_ids.len()).map_err(|_| JointBudgetError::ArithmeticOverflow)?)
        .ok_or(JointBudgetError::ArithmeticOverflow)?;
    if runs.len() != expected {
        return Err(JointBudgetError::IncompleteEnumeration {
            expected,
            actual: runs.len(),
        });
    }

    let item_index: BTreeMap<u64, usize> = sorted_registry
        .iter()
        .enumerate()
        .map(|(index, definition)| (definition.action_id, index))
        .collect();
    let conflict_groups: Vec<u64> = sorted_registry.iter().map(|definition| definition.conflict_group).collect();
    let registry_digest = action_registry_sha256(&sorted_registry)?;

    let mut by_mask = BTreeMap::<u64, (JointSubsetSummary, bool)>::new();
    let mut context_reference: Option<&JointBudgetContext> = None;
    let mut run_ids = BTreeSet::new();

    for run in runs {
        run.context.validate()?;
        if run.context.action_registry_sha256 != registry_digest {
            return Err(JointBudgetError::ContextMismatch);
        }
        if run.run_id == 0 {
            return Err(JointBudgetError::DuplicateRunId { run_id: 0 });
        }
        if let Some(reference) = context_reference {
            if reference != &run.context {
                return Err(JointBudgetError::ContextMismatch);
            }
        } else {
            context_reference = Some(&run.context);
        }
        if !run_ids.insert(run.run_id) {
            return Err(JointBudgetError::DuplicateRunId { run_id: run.run_id });
        }
        run.outcome.validate(run.context.total_exposure_us)?;

        let mut selected = BTreeSet::new();
        let mut mask = 0_u64;
        for action_id in &run.selected_action_ids {
            if !selected.insert(*action_id) {
                return Err(JointBudgetError::UnsortedOrDuplicateSelection);
            }
            let index = *item_index
                .get(action_id)
                .ok_or(JointBudgetError::UnknownActionId { action_id: *action_id })?;
            mask |= 1_u64 << index;
        }
        if run.selected_action_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(JointBudgetError::UnsortedOrDuplicateSelection);
        }

        let summary = JointSubsetSummary {
            subset_mask: mask,
            selected_action_ids: run.selected_action_ids.clone(),
            measured_resources: run.measured_resources,
            outcome: run.outcome,
        };
        let mut group_counts = BTreeMap::<u64, u8>::new();
        let mut compatibility_valid = true;
        for action_id in &run.selected_action_ids {
            let index = *item_index
                .get(action_id)
                .ok_or(JointBudgetError::UnknownActionId { action_id: *action_id })?;
            let group = conflict_groups[index];
            if group != 0 {
                let count = group_counts.entry(group).or_insert(0);
                *count = count.checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
                if *count > 1 {
                    compatibility_valid = false;
                }
            }
        }
        let feasible = compatibility_valid
            && run.measured_resources.fits_within(run.context.resource_budget);
        if by_mask.insert(mask, (summary, feasible)).is_some() {
            return Err(JointBudgetError::DuplicateSubset { mask });
        }
    }

    for mask in 0..u64::try_from(expected).map_err(|_| JointBudgetError::ArithmeticOverflow)? {
        if !by_mask.contains_key(&mask) {
            return Err(JointBudgetError::MissingSubset { mask });
        }
    }

    let feasible: Vec<JointSubsetSummary> = by_mask
        .values()
        .filter_map(|(summary, is_feasible)| if *is_feasible { Some(summary.clone()) } else { None })
        .collect();
    if feasible.is_empty() {
        return Err(JointBudgetError::NoFeasibleSubset);
    }

    let candidate_mask = mask_for_actions(candidate_selected_action_ids, &item_index)?;
    let (candidate, candidate_feasible) = by_mask
        .get(&candidate_mask)
        .ok_or(JointBudgetError::CandidateSubsetMissing)?;
    if !candidate_feasible {
        return Err(JointBudgetError::CandidateSubsetInfeasible);
    }

    let best_wrong_belief = choose_minimum(&feasible, |outcome| outcome.wrong_belief_us);
    let best_abstention = choose_minimum(&feasible, |outcome| outcome.abstention_us);
    let best_missing_output = choose_minimum(&feasible, |outcome| outcome.missing_output_us);
    let pareto_frontier = pareto_frontier(&feasible);

    Ok(JointBudgetQualification {
        candidate_regret: JointExposureRegret {
            wrong_belief_us: candidate.outcome.wrong_belief_us.saturating_sub(best_wrong_belief.outcome.wrong_belief_us),
            abstention_us: candidate.outcome.abstention_us.saturating_sub(best_abstention.outcome.abstention_us),
            missing_output_us: candidate.outcome.missing_output_us.saturating_sub(best_missing_output.outcome.missing_output_us),
        },
        pareto_frontier,
        best_wrong_belief,
        best_abstention,
        best_missing_output,
        evaluated_subset_count: by_mask.len(),
        feasible_subset_count: feasible.len(),
    })
}

fn mask_for_actions(
    selected_action_ids: &[u64],
    item_index: &BTreeMap<u64, usize>,
) -> Result<u64, JointBudgetError> {
    if selected_action_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(JointBudgetError::UnsortedOrDuplicateSelection);
    }
    let mut mask = 0_u64;
    for action_id in selected_action_ids {
        let index = *item_index
            .get(action_id)
            .ok_or(JointBudgetError::UnknownActionId { action_id: *action_id })?;
        mask |= 1_u64 << index;
    }
    Ok(mask)
}

fn choose_minimum(
    summaries: &[JointSubsetSummary],
    metric: impl Fn(JointExposure) -> u64,
) -> JointSubsetSummary {
    summaries
        .iter()
        .min_by(|left, right| {
            metric(left.outcome)
                .cmp(&metric(right.outcome))
                .then_with(|| left.selected_action_ids.cmp(&right.selected_action_ids))
        })
        .expect("feasible subset collection is non-empty")
        .clone()
}

fn pareto_frontier(summaries: &[JointSubsetSummary]) -> Vec<JointSubsetSummary> {
    let mut frontier = summaries
        .iter()
        .filter(|candidate| {
            !summaries.iter().any(|other| {
                other.subset_mask != candidate.subset_mask && dominates(other, candidate)
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        left.outcome.wrong_belief_us.cmp(&right.outcome.wrong_belief_us)
            .then_with(|| left.outcome.abstention_us.cmp(&right.outcome.abstention_us))
            .then_with(|| left.outcome.missing_output_us.cmp(&right.outcome.missing_output_us))
            .then_with(|| left.selected_action_ids.cmp(&right.selected_action_ids))
    });
    frontier
}

fn dominates(left: &JointSubsetSummary, right: &JointSubsetSummary) -> bool {
    let a = left.outcome;
    let b = right.outcome;
    let x = left.measured_resources;
    let y = right.measured_resources;
    let no_worse = a.wrong_belief_us <= b.wrong_belief_us
        && a.abstention_us <= b.abstention_us
        && a.missing_output_us <= b.missing_output_us
        && x.persistent_bytes <= y.persistent_bytes
        && x.retained_byte_us <= y.retained_byte_us
        && x.io_bytes <= y.io_bytes
        && x.latency_us <= y.latency_us
        && x.compute_units <= y.compute_units
        && x.observations <= y.observations
        && x.energy_nj <= y.energy_nj
        && x.verification_units <= y.verification_units;
    let strictly_better = a.wrong_belief_us < b.wrong_belief_us
        || a.abstention_us < b.abstention_us
        || a.missing_output_us < b.missing_output_us
        || x.persistent_bytes < y.persistent_bytes
        || x.retained_byte_us < y.retained_byte_us
        || x.io_bytes < y.io_bytes
        || x.latency_us < y.latency_us
        || x.compute_units < y.compute_units
        || x.observations < y.observations
        || x.energy_nj < y.energy_nj
        || x.verification_units < y.verification_units;
    no_worse && strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(items: &[(u64, u64)]) -> Vec<JointActionDefinition> {
        items.iter().map(|(action_id, conflict_group)| {
            let mut action_semantics_sha256 = [0_u8; 32];
            action_semantics_sha256[..8].copy_from_slice(&action_id.to_le_bytes());
            JointActionDefinition {
                action_id: *action_id,
                conflict_group: *conflict_group,
                action_semantics_sha256,
            }
        }).collect()
    }

    fn context() -> JointBudgetContext {
        let action_registry_sha256 = action_registry_sha256(&registry(&[(10, 0), (20, 0)])).unwrap();
        JointBudgetContext {
            episode_id: 77,
            task_sha256: [1; 32],
            future_stream_sha256: [2; 32],
            model_sha256: [3; 32],
            initial_state_sha256: [4; 32],
            candidate_feature_sha256: [5; 32],
            action_registry_sha256,
            total_exposure_us: 100,
            resource_budget: JointResourceUse {
                persistent_bytes: 100,
                retained_byte_us: 10_000,
                io_bytes: 100,
                latency_us: 10_000,
                compute_units: 100,
                observations: 100,
                energy_nj: 1_000,
                verification_units: 10,
            },
        }
    }

    fn measured(observations: u64, compute: u64) -> JointResourceUse {
        JointResourceUse {
            persistent_bytes: 10,
            retained_byte_us: 100,
            io_bytes: 20,
            latency_us: 10,
            compute_units: compute,
            observations,
            energy_nj: 50,
            verification_units: 1,
        }
    }

    fn run(
        run_id: u64,
        selected_action_ids: Vec<u64>,
        resources: JointResourceUse,
        outcome: JointExposure,
    ) -> JointSubsetRun {
        JointSubsetRun {
            context: context(),
            run_id,
            selected_action_ids,
            measured_resources: resources,
            outcome,
        }
    }

    fn all_four_runs() -> Vec<JointSubsetRun> {
        vec![
            run(1, vec![], measured(0, 0), JointExposure { wrong_belief_us: 10, abstention_us: 8, missing_output_us: 7 }),
            run(2, vec![10], measured(2, 2), JointExposure { wrong_belief_us: 4, abstention_us: 8, missing_output_us: 5 }),
            run(3, vec![20], measured(3, 3), JointExposure { wrong_belief_us: 9, abstention_us: 2, missing_output_us: 4 }),
            run(4, vec![10, 20], measured(5, 5), JointExposure { wrong_belief_us: 3, abstention_us: 3, missing_output_us: 1 }),
        ]
    }

    #[test]
    fn exact_joint_oracle_keeps_componentwise_optima_and_regret_separate() {
        let result = qualify_joint_subsets(&registry(&[(20, 0), (10, 0)]), &all_four_runs(), &[10]).unwrap();
        assert_eq!(result.evaluated_subset_count, 4);
        assert_eq!(result.feasible_subset_count, 4);
        assert_eq!(result.best_wrong_belief.selected_action_ids, vec![10, 20]);
        assert_eq!(result.best_abstention.selected_action_ids, vec![20]);
        assert_eq!(result.best_missing_output.selected_action_ids, vec![10, 20]);
        assert_eq!(result.candidate_regret, JointExposureRegret {
            wrong_belief_us: 1,
            abstention_us: 6,
            missing_output_us: 4,
        });
    }

    #[test]
    fn infeasible_subsets_are_excluded_from_the_oracle_frontier() {
        let mut runs = all_four_runs();
        runs[3].measured_resources.observations = 101;
        let result = qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]).unwrap();
        assert_eq!(result.evaluated_subset_count, 4);
        assert_eq!(result.feasible_subset_count, 3);
        assert!(!result.pareto_frontier.iter().any(|entry| entry.selected_action_ids == vec![10, 20]));
    }

    #[test]
    fn candidate_subset_over_budget_is_not_qualified() {
        let mut runs = all_four_runs();
        runs[3].measured_resources.observations = 101;
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10, 20]),
            Err(JointBudgetError::CandidateSubsetInfeasible)
        );
    }

    #[test]
    fn incomplete_subset_enumeration_is_rejected() {
        let runs = vec![all_four_runs()[0].clone()];
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[]),
            Err(JointBudgetError::IncompleteEnumeration { expected: 4, actual: 1 })
        );
    }

    #[test]
    fn future_stream_or_resource_context_mismatch_is_rejected() {
        let mut runs = all_four_runs();
        runs[2].context.future_stream_sha256 = [9; 32];
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::ContextMismatch)
        );
    }

    #[test]
    fn duplicate_run_ids_are_rejected() {
        let mut runs = all_four_runs();
        runs[1].run_id = runs[0].run_id;
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::DuplicateRunId { run_id: 1 })
        );
    }

    #[test]
    fn unknown_action_and_unsorted_selection_are_rejected() {
        let mut runs = all_four_runs();
        runs[3].selected_action_ids = vec![20, 10];
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::UnsortedOrDuplicateSelection)
        );

        let mut runs = all_four_runs();
        runs[3].selected_action_ids = vec![10, 30];
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::UnknownActionId { action_id: 30 })
        );
    }

    #[test]
    fn exposure_cannot_exceed_episode_horizon() {
        let mut runs = all_four_runs();
        runs[0].outcome = JointExposure { wrong_belief_us: 80, abstention_us: 20, missing_output_us: 1 };
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::ExposureExceedsEpisode { measured_us: 101, total_us: 100 })
        );
    }

    #[test]
    fn pareto_frontier_preserves_quality_cost_tradeoffs() {
        let mut runs = all_four_runs();
        runs[0].outcome = JointExposure { wrong_belief_us: 8, abstention_us: 8, missing_output_us: 8 };
        runs[0].measured_resources = measured(1, 1);
        runs[1].outcome = JointExposure { wrong_belief_us: 2, abstention_us: 7, missing_output_us: 7 };
        runs[1].measured_resources = measured(5, 5);
        runs[2].outcome = JointExposure { wrong_belief_us: 7, abstention_us: 2, missing_output_us: 7 };
        runs[2].measured_resources = measured(6, 6);
        runs[3].outcome = JointExposure { wrong_belief_us: 1, abstention_us: 1, missing_output_us: 1 };
        runs[3].measured_resources = measured(10, 10);
        let result = qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]).unwrap();
        assert!(result.pareto_frontier.len() >= 2);
    }

    #[test]
    fn zero_action_or_duplicate_registry_fails_closed() {
        let runs = vec![run(1, vec![], measured(0, 0), JointExposure::default())];
        assert_eq!(qualify_joint_subsets(&registry(&[(0, 0)]), &runs, &[]), Err(JointBudgetError::ZeroActionId));
        assert!(matches!(
            qualify_joint_subsets(&[JointActionDefinition { action_id: 1, conflict_group: 0 }, JointActionDefinition { action_id: 1, conflict_group: 0 }], &runs, &[]),
            Err(JointBudgetError::DuplicateActionId { action_id: 1 })
        ));
    }

    #[test]
    fn mutually_exclusive_action_subsets_are_not_oracle_feasible() {
        let mut runs = all_four_runs();
        let conflicting_registry = registry(&[(10, 9), (20, 9)]);
        let digest = action_registry_sha256(&conflicting_registry).unwrap();
        for run in &mut runs {
            run.context.action_registry_sha256 = digest;
        }
        let result = qualify_joint_subsets(&conflicting_registry, &runs, &[10]).unwrap();
        assert_eq!(result.evaluated_subset_count, 4);
        assert_eq!(result.feasible_subset_count, 3);
        assert!(!result.pareto_frontier.iter().any(|entry| entry.selected_action_ids == vec![10, 20]));
        assert_eq!(
            qualify_joint_subsets(&conflicting_registry, &runs, &[10, 20]),
            Err(JointBudgetError::CandidateSubsetInfeasible)
        );
    }

    #[test]
    fn action_registry_commitment_and_context_identity_are_required() {
        let mut runs = all_four_runs();
        runs[1].context.action_registry_sha256 = [8; 32];
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::ContextMismatch)
        );
        let mut runs = all_four_runs();
        runs[0].context.action_registry_sha256 = [0; 32];
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[10]),
            Err(JointBudgetError::InvalidContextCommitment)
        );
    }

    #[test]
    fn registry_digest_binds_ids_groups_and_semantic_contracts() {
        let base = action_registry_sha256(&registry(&[(10, 0), (20, 0)])).unwrap();
        let reordered = action_registry_sha256(&registry(&[(20, 0), (10, 0)])).unwrap();
        let constrained = action_registry_sha256(&registry(&[(10, 9), (20, 9)])).unwrap();
        let mut changed_semantics = registry(&[(10, 0), (20, 0)]);
        changed_semantics[0].action_semantics_sha256 = [99; 32];
        let changed_contract = action_registry_sha256(&changed_semantics).unwrap();
        assert_eq!(base, reordered);
        assert_ne!(base, constrained);
        assert_ne!(base, changed_contract);
    }

    #[test]
    fn zero_action_semantics_digest_is_rejected() {
        let mut invalid = registry(&[(10, 0), (20, 0)]);
        invalid[0].action_semantics_sha256 = [0; 32];
        assert_eq!(
            action_registry_sha256(&invalid),
            Err(JointBudgetError::InvalidActionSemanticsDigest { action_id: 10 })
        );
    }

    #[test]
    fn no_feasible_subset_is_an_explicit_failure() {
        let mut runs = all_four_runs();
        for run in &mut runs {
            run.context.resource_budget.observations = 0;
            run.measured_resources.observations = 1;
        }
        assert_eq!(
            qualify_joint_subsets(&registry(&[(10, 0), (20, 0)]), &runs, &[]),
            Err(JointBudgetError::NoFeasibleSubset)
        );
    }
}
