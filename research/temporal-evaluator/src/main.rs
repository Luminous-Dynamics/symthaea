//! Evaluator-side counterfactual retention primitives.
//!
//! This is intentionally a separate package from the candidate benchmark
//! crate. It contains evaluator-side outcome records and must not be linked
//! into or executed by an untrusted candidate. Package separation is not an
//! OS security boundary: #9437 still requires process/UID isolation, protected
//! storage, evaluator-owned signing keys and replay-resistant receipts.
//!
//! No function here supplies evaluator outcomes to candidate-side policies.

pub mod witness;
pub mod joint_budget;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAX_EXHAUSTIVE_ITEMS: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpisodeContext {
    pub episode_id: u64,
    pub task_sha256: [u8; 32],
    pub future_stream_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub initial_state_sha256: [u8; 32],
    /// Hash of the candidate-visible feature snapshot frozen before treatment.
    pub candidate_feature_sha256: [u8; 32],
    pub total_exposure_us: u64,
    pub byte_budget: u64,
    pub compute_budget: u64,
    pub refresh_budget: u64,
    pub observation_budget: u64,
    pub energy_budget_nj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryItem {
    pub item_id: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExposureOutcome {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

impl ExposureOutcome {
    fn validate(self, total_exposure_us: u64) -> Result<(), CounterfactualError> {
        let total = self
            .wrong_belief_us
            .checked_add(self.abstention_us)
            .and_then(|value| value.checked_add(self.missing_output_us))
            .ok_or(CounterfactualError::ArithmeticOverflow)?;
        if total > total_exposure_us {
            return Err(CounterfactualError::ExposureExceedsEpisode {
                exposure_us: total,
                episode_us: total_exposure_us,
            });
        }
        Ok(())
    }
}

/// All three deltas are kept separate. Positive means retaining the treatment
/// set reduced that exposure relative to evicting it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExposureDelta {
    pub wrong_belief_us: i128,
    pub abstention_us: i128,
    pub missing_output_us: i128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredRunCost {
    pub compute_used: u64,
    pub refreshes_used: u64,
    pub observations_used: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
    /// Calibrated evaluator measurement, absent when unavailable.
    pub measured_energy_nj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterfactualRun {
    pub context: EpisodeContext,
    pub run_id: u64,
    /// Sorted, unique IDs of items retained during this run.
    pub selected_item_ids: Vec<u64>,
    pub bytes_used: u64,
    /// Evaluator-measured retained-memory occupancy integrated over time.
    pub retained_byte_us: u128,
    pub measured_cost: MeasuredRunCost,
    pub outcome: ExposureOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TreatmentOrder {
    RetainFirst,
    EvictFirst,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterfactualEffect {
    pub episode_id: u64,
    /// Reproducible evaluator-side order randomization; not a cryptographic RNG.
    pub first_treatment: TreatmentOrder,
    pub intervention_item_ids: Vec<u64>,
    pub retained_run_id: u64,
    pub evicted_run_id: u64,
    pub retained_bytes_used: u64,
    pub evicted_bytes_used: u64,
    pub retained_byte_us: u128,
    pub evicted_byte_us: u128,
    pub retained_cost: MeasuredRunCost,
    pub evicted_cost: MeasuredRunCost,
    pub delta_evict_minus_retain: ExposureDelta,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubsetSummary {
    pub subset_mask: u64,
    pub selected_item_ids: Vec<u64>,
    pub bytes_used: u64,
    /// Evaluator-measured retained-memory occupancy integrated over time.
    pub retained_byte_us: u128,
    pub measured_cost: MeasuredRunCost,
    pub outcome: ExposureOutcome,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubsetRegret {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactSubsetQualification {
    /// Nondominated subsets. No weighted scalar objective is applied.
    pub pareto_frontier: Vec<SubsetSummary>,
    /// Independent evaluator-only optimum for each exposure component.
    pub best_wrong_belief: SubsetSummary,
    pub best_abstention: SubsetSummary,
    pub best_missing_output: SubsetSummary,
    pub candidate_regret: SubsetRegret,
    pub evaluated_subset_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CounterfactualError {
    EmptyIntervention,
    TooManyItems { requested: usize, maximum: usize },
    DuplicateItemId { item_id: u64 },
    ZeroItemSize { item_id: u64 },
    UnsortedOrDuplicateSelection,
    UnknownSelectedItem { item_id: u64 },
    ContextMismatch,
    RunIdCollision { run_id: u64 },
    RetainedTreatmentMissingItem { item_id: u64 },
    EvictedTreatmentStillContainsItem { item_id: u64 },
    TreatmentDiffersByUnexpectedItems,
    ByteCountMismatch { reported: u64, computed: u64 },
    OutOfBudget { used: u64, budget: u64 },
    ComputeOutOfBudget { used: u64, budget: u64 },
    RefreshOutOfBudget { used: u64, budget: u64 },
    ObservationOutOfBudget { used: u64, budget: u64 },
    EnergyMeasurementMissing,
    EnergyOutOfBudget { used: u64, budget: u64 },
    ExposureExceedsEpisode { exposure_us: u64, episode_us: u64 },
    IncompleteSubsetEnumeration { expected: usize, actual: usize },
    DuplicateSubset { mask: u64 },
    MissingSubset { mask: u64 },
    CandidateSubsetMissing,
    ArithmeticOverflow,
}

impl fmt::Display for CounterfactualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIntervention => write!(f, "counterfactual intervention set must not be empty"),
            Self::TooManyItems { requested, maximum } => write!(
                f,
                "exhaustive subset evaluation supports at most {maximum} items, received {requested}"
            ),
            Self::DuplicateItemId { item_id } => write!(f, "duplicate item ID {item_id}"),
            Self::ZeroItemSize { item_id } => write!(f, "item {item_id} has zero bytes"),
            Self::UnsortedOrDuplicateSelection => {
                write!(f, "selected item IDs must be strictly ascending and unique")
            }
            Self::UnknownSelectedItem { item_id } => {
                write!(f, "selected item ID {item_id} is not in the episode registry")
            }
            Self::ContextMismatch => write!(
                f,
                "paired runs do not share task, future stream, model, initial state, feature snapshot and resource budgets"
            ),
            Self::RunIdCollision { run_id } => write!(f, "duplicate evaluator run ID {run_id}"),
            Self::RetainedTreatmentMissingItem { item_id } => {
                write!(f, "retained treatment does not contain item {item_id}")
            }
            Self::EvictedTreatmentStillContainsItem { item_id } => {
                write!(f, "evicted treatment still contains item {item_id}")
            }
            Self::TreatmentDiffersByUnexpectedItems => {
                write!(f, "paired treatment runs differ by items outside the declared intervention set")
            }
            Self::ByteCountMismatch { reported, computed } => write!(
                f,
                "reported memory bytes {reported} differ from selected-item sum {computed}"
            ),
            Self::OutOfBudget { used, budget } => {
                write!(f, "selected bytes {used} exceed byte budget {budget}")
            }
            Self::ComputeOutOfBudget { used, budget } => {
                write!(f, "compute use {used} exceeds compute budget {budget}")
            }
            Self::RefreshOutOfBudget { used, budget } => {
                write!(f, "refresh count {used} exceeds refresh budget {budget}")
            }
            Self::ObservationOutOfBudget { used, budget } => {
                write!(f, "observation count {used} exceeds observation budget {budget}")
            }
            Self::EnergyMeasurementMissing => {
                write!(f, "episode declares an energy budget but calibrated energy measurement is absent")
            }
            Self::EnergyOutOfBudget { used, budget } => {
                write!(f, "measured energy {used} nJ exceeds budget {budget} nJ")
            }
            Self::ExposureExceedsEpisode { exposure_us, episode_us } => write!(
                f,
                "wrong/abstention/missing exposure {exposure_us} exceeds episode duration {episode_us}"
            ),
            Self::IncompleteSubsetEnumeration { expected, actual } => write!(
                f,
                "exact subset oracle requires {expected} records, received {actual}"
            ),
            Self::DuplicateSubset { mask } => write!(f, "duplicate subset mask {mask:#x}"),
            Self::MissingSubset { mask } => write!(f, "missing subset mask {mask:#x}"),
            Self::CandidateSubsetMissing => write!(f, "candidate subset is absent from evaluator enumeration"),
            Self::ArithmeticOverflow => write!(f, "counterfactual arithmetic overflowed"),
        }
    }
}

impl std::error::Error for CounterfactualError {}

/// Evaluate one paired retain/evict intervention. The runs must be identical
/// in all declared context fields and differ only in the intervention IDs.
pub fn evaluate_paired_intervention(
    items: &[MemoryItem],
    retained: &CounterfactualRun,
    evicted: &CounterfactualRun,
    intervention_item_ids: &[u64],
    evaluator_randomization_seed: u64,
) -> Result<CounterfactualEffect, CounterfactualError> {
    if intervention_item_ids.is_empty() {
        return Err(CounterfactualError::EmptyIntervention);
    }
    validate_item_registry(items)?;
    validate_run(items, retained)?;
    validate_run(items, evicted)?;
    validate_same_context(retained, evicted)?;

    let intervention = sorted_unique_set(intervention_item_ids)?;
    let retained_set = sorted_unique_set(&retained.selected_item_ids)?;
    let evicted_set = sorted_unique_set(&evicted.selected_item_ids)?;

    for item_id in &intervention {
        if !retained_set.contains(item_id) {
            return Err(CounterfactualError::RetainedTreatmentMissingItem { item_id: *item_id });
        }
        if evicted_set.contains(item_id) {
            return Err(CounterfactualError::EvictedTreatmentStillContainsItem { item_id: *item_id });
        }
    }

    let retained_without_intervention: BTreeSet<_> =
        retained_set.difference(&intervention).copied().collect();
    if retained_without_intervention != evicted_set {
        return Err(CounterfactualError::TreatmentDiffersByUnexpectedItems);
    }

    let intervention_item_ids: Vec<_> = intervention.into_iter().collect();
    let first_treatment = randomize_treatment_order(
        evaluator_randomization_seed,
        retained.context.episode_id,
        &intervention_item_ids,
    )?;

    Ok(CounterfactualEffect {
        episode_id: retained.context.episode_id,
        first_treatment,
        intervention_item_ids,
        retained_run_id: retained.run_id,
        evicted_run_id: evicted.run_id,
        retained_bytes_used: retained.bytes_used,
        evicted_bytes_used: evicted.bytes_used,
        retained_byte_us: retained.retained_byte_us,
        evicted_byte_us: evicted.retained_byte_us,
        retained_cost: retained.measured_cost,
        evicted_cost: evicted.measured_cost,
        delta_evict_minus_retain: ExposureDelta {
            wrong_belief_us: i128::from(evicted.outcome.wrong_belief_us)
                - i128::from(retained.outcome.wrong_belief_us),
            abstention_us: i128::from(evicted.outcome.abstention_us)
                - i128::from(retained.outcome.abstention_us),
            missing_output_us: i128::from(evicted.outcome.missing_output_us)
                - i128::from(retained.outcome.missing_output_us),
        },
    })
}

/// Deterministic evaluator-side treatment-order randomization for paired
/// interventions. The seed must remain evaluator-owned. SplitMix64 provides
/// reproducible ordering only; it is not a cryptographic secrecy mechanism.
pub fn randomize_treatment_order(
    evaluator_seed: u64,
    episode_id: u64,
    intervention_item_ids: &[u64],
) -> Result<TreatmentOrder, CounterfactualError> {
    if intervention_item_ids.is_empty() {
        return Err(CounterfactualError::EmptyIntervention);
    }
    let ids = sorted_unique_set(intervention_item_ids)?;
    let mut state = evaluator_seed ^ episode_id.rotate_left(23) ^ 0x5445_4D50_4F52_414C;
    for item_id in ids {
        state = mix64(state ^ item_id.rotate_left(11));
    }
    Ok(if mix64(state) & 1 == 0 {
        TreatmentOrder::RetainFirst
    } else {
        TreatmentOrder::EvictFirst
    })
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Exhaustively evaluate all subsets for at most 16 items. Each supplied run
/// must use the same task, future stream, model, initial state, candidate
/// feature snapshot and resource envelope. This is an evaluator-only
/// clairvoyant upper bound, never a candidate policy.
pub fn qualify_exact_subsets(
    items: &[MemoryItem],
    runs: &[CounterfactualRun],
    candidate_selected_item_ids: &[u64],
) -> Result<ExactSubsetQualification, CounterfactualError> {
    validate_item_registry(items)?;
    if items.len() > MAX_EXHAUSTIVE_ITEMS {
        return Err(CounterfactualError::TooManyItems {
            requested: items.len(),
            maximum: MAX_EXHAUSTIVE_ITEMS,
        });
    }
    let expected = 1_usize
        .checked_shl(u32::try_from(items.len()).map_err(|_| CounterfactualError::ArithmeticOverflow)?)
        .ok_or(CounterfactualError::ArithmeticOverflow)?;
    if runs.len() != expected {
        return Err(CounterfactualError::IncompleteSubsetEnumeration {
            expected,
            actual: runs.len(),
        });
    }

    let mut ordered_items = items.to_vec();
    ordered_items.sort_by_key(|item| item.item_id);
    let item_index: BTreeMap<u64, usize> = ordered_items
        .iter()
        .enumerate()
        .map(|(index, item)| (item.item_id, index))
        .collect();

    let mut by_mask = BTreeMap::<u64, SubsetSummary>::new();
    let mut context_reference: Option<&EpisodeContext> = None;
    let mut run_ids = BTreeSet::new();

    for run in runs {
        validate_run(items, run)?;
        if let Some(reference) = context_reference {
            if reference != &run.context {
                return Err(CounterfactualError::ContextMismatch);
            }
        } else {
            context_reference = Some(&run.context);
        }
        if !run_ids.insert(run.run_id) {
            return Err(CounterfactualError::RunIdCollision { run_id: run.run_id });
        }

        let mut mask = 0_u64;
        for item_id in &run.selected_item_ids {
            let index = *item_index
                .get(item_id)
                .ok_or(CounterfactualError::UnknownSelectedItem { item_id: *item_id })?;
            mask |= 1_u64 << index;
        }
        let summary = SubsetSummary {
            subset_mask: mask,
            selected_item_ids: run.selected_item_ids.clone(),
            bytes_used: run.bytes_used,
            retained_byte_us: run.retained_byte_us,
            measured_cost: run.measured_cost,
            outcome: run.outcome,
        };
        if by_mask.insert(mask, summary).is_some() {
            return Err(CounterfactualError::DuplicateSubset { mask });
        }
    }

    for mask in 0..u64::try_from(expected).map_err(|_| CounterfactualError::ArithmeticOverflow)? {
        if !by_mask.contains_key(&mask) {
            return Err(CounterfactualError::MissingSubset { mask });
        }
    }

    let candidate_set = sorted_unique_set(candidate_selected_item_ids)?;
    let candidate_mask = mask_for_set(&candidate_set, &item_index)?;
    let candidate = by_mask
        .get(&candidate_mask)
        .ok_or(CounterfactualError::CandidateSubsetMissing)?;

    let summaries: Vec<_> = by_mask.values().cloned().collect();
    let best_wrong_belief = choose_minimum(&summaries, |o| o.wrong_belief_us);
    let best_abstention = choose_minimum(&summaries, |o| o.abstention_us);
    let best_missing_output = choose_minimum(&summaries, |o| o.missing_output_us);
    let pareto_frontier = pareto_frontier(&summaries);

    Ok(ExactSubsetQualification {
        candidate_regret: SubsetRegret {
            wrong_belief_us: candidate.outcome.wrong_belief_us
                .saturating_sub(best_wrong_belief.outcome.wrong_belief_us),
            abstention_us: candidate.outcome.abstention_us
                .saturating_sub(best_abstention.outcome.abstention_us),
            missing_output_us: candidate.outcome.missing_output_us
                .saturating_sub(best_missing_output.outcome.missing_output_us),
        },
        pareto_frontier,
        best_wrong_belief,
        best_abstention,
        best_missing_output,
        evaluated_subset_count: summaries.len(),
    })
}

fn validate_item_registry(items: &[MemoryItem]) -> Result<(), CounterfactualError> {
    let mut seen = BTreeSet::new();
    for item in items {
        if !seen.insert(item.item_id) {
            return Err(CounterfactualError::DuplicateItemId { item_id: item.item_id });
        }
        if item.bytes == 0 {
            return Err(CounterfactualError::ZeroItemSize { item_id: item.item_id });
        }
    }
    Ok(())
}

fn validate_run(
    items: &[MemoryItem],
    run: &CounterfactualRun,
) -> Result<(), CounterfactualError> {
    let selected = sorted_unique_set(&run.selected_item_ids)?;
    let by_id: BTreeMap<_, _> = items.iter().map(|item| (item.item_id, item.bytes)).collect();
    let mut computed = 0_u64;
    for item_id in &selected {
        let bytes = *by_id
            .get(item_id)
            .ok_or(CounterfactualError::UnknownSelectedItem { item_id: *item_id })?;
        computed = computed
            .checked_add(bytes)
            .ok_or(CounterfactualError::ArithmeticOverflow)?;
    }
    if run.bytes_used != computed {
        return Err(CounterfactualError::ByteCountMismatch {
            reported: run.bytes_used,
            computed,
        });
    }
    if run.bytes_used > run.context.byte_budget {
        return Err(CounterfactualError::OutOfBudget {
            used: run.bytes_used,
            budget: run.context.byte_budget,
        });
    }
    if run.measured_cost.compute_used > run.context.compute_budget {
        return Err(CounterfactualError::ComputeOutOfBudget {
            used: run.measured_cost.compute_used,
            budget: run.context.compute_budget,
        });
    }
    if run.measured_cost.refreshes_used > run.context.refresh_budget {
        return Err(CounterfactualError::RefreshOutOfBudget {
            used: run.measured_cost.refreshes_used,
            budget: run.context.refresh_budget,
        });
    }
    if run.measured_cost.observations_used > run.context.observation_budget {
        return Err(CounterfactualError::ObservationOutOfBudget {
            used: run.measured_cost.observations_used,
            budget: run.context.observation_budget,
        });
    }
    match (run.context.energy_budget_nj, run.measured_cost.measured_energy_nj) {
        (Some(_), None) => return Err(CounterfactualError::EnergyMeasurementMissing),
        (Some(budget), Some(used)) if used > budget => {
            return Err(CounterfactualError::EnergyOutOfBudget { used, budget });
        }
        _ => {}
    }
    run.outcome.validate(run.context.total_exposure_us)?;
    Ok(())
}

fn validate_same_context(
    left: &CounterfactualRun,
    right: &CounterfactualRun,
) -> Result<(), CounterfactualError> {
    if left.context != right.context {
        return Err(CounterfactualError::ContextMismatch);
    }
    if left.run_id == right.run_id {
        return Err(CounterfactualError::RunIdCollision { run_id: left.run_id });
    }
    Ok(())
}

fn sorted_unique_set(values: &[u64]) -> Result<BTreeSet<u64>, CounterfactualError> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CounterfactualError::UnsortedOrDuplicateSelection);
    }
    Ok(values.iter().copied().collect())
}

fn mask_for_set(
    values: &BTreeSet<u64>,
    item_index: &BTreeMap<u64, usize>,
) -> Result<u64, CounterfactualError> {
    let mut mask = 0_u64;
    for item_id in values {
        let index = *item_index
            .get(item_id)
            .ok_or(CounterfactualError::UnknownSelectedItem { item_id: *item_id })?;
        mask |= 1_u64 << index;
    }
    Ok(mask)
}

fn choose_minimum(
    summaries: &[SubsetSummary],
    selector: impl Fn(ExposureOutcome) -> u64,
) -> SubsetSummary {
    summaries
        .iter()
        .min_by(|left, right| {
            selector(left.outcome)
                .cmp(&selector(right.outcome))
                .then_with(|| left.bytes_used.cmp(&right.bytes_used))
                .then_with(|| left.selected_item_ids.cmp(&right.selected_item_ids))
        })
        .expect("exact subset enumeration is non-empty")
        .clone()
}

fn pareto_frontier(summaries: &[SubsetSummary]) -> Vec<SubsetSummary> {
    let mut frontier = summaries
        .iter()
        .filter(|candidate| {
            !summaries.iter().any(|other| {
                other.subset_mask != candidate.subset_mask
                    && dominates(other.outcome, candidate.outcome)
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| {
        left.outcome
            .wrong_belief_us
            .cmp(&right.outcome.wrong_belief_us)
            .then_with(|| left.outcome.abstention_us.cmp(&right.outcome.abstention_us))
            .then_with(|| left.outcome.missing_output_us.cmp(&right.outcome.missing_output_us))
            .then_with(|| left.bytes_used.cmp(&right.bytes_used))
            .then_with(|| left.selected_item_ids.cmp(&right.selected_item_ids))
    });
    frontier
}

fn dominates(left: ExposureOutcome, right: ExposureOutcome) -> bool {
    let no_worse = left.wrong_belief_us <= right.wrong_belief_us
        && left.abstention_us <= right.abstention_us
        && left.missing_output_us <= right.missing_output_us;
    let strictly_better = left.wrong_belief_us < right.wrong_belief_us
        || left.abstention_us < right.abstention_us
        || left.missing_output_us < right.missing_output_us;
    no_worse && strictly_better
}

fn main() {
    println!(
        "Evaluator-only counterfactual primitives. No candidate, hidden task or oracle data was loaded."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> EpisodeContext {
        EpisodeContext {
            episode_id: 42,
            task_sha256: [1; 32],
            future_stream_sha256: [2; 32],
            model_sha256: [3; 32],
            initial_state_sha256: [4; 32],
            candidate_feature_sha256: [5; 32],
            total_exposure_us: 100,
            byte_budget: 10,
            compute_budget: 1_000,
            refresh_budget: 20,
            observation_budget: 30,
            energy_budget_nj: Some(10_000),
        }
    }

    fn run(
        run_id: u64,
        selected_item_ids: Vec<u64>,
        bytes_used: u64,
        outcome: ExposureOutcome,
    ) -> CounterfactualRun {
        CounterfactualRun {
            context: context(),
            run_id,
            selected_item_ids,
            bytes_used,
            retained_byte_us: u128::from(bytes_used) * 100,
            measured_cost: MeasuredRunCost {
                compute_used: 500,
                refreshes_used: 3,
                observations_used: 10,
                read_bytes: 512,
                write_bytes: 128,
                measured_energy_nj: Some(2_000),
            },
            outcome,
        }
    }

    fn registry() -> Vec<MemoryItem> {
        vec![
            MemoryItem { item_id: 10, bytes: 2 },
            MemoryItem { item_id: 20, bytes: 3 },
        ]
    }

    #[test]
    fn paired_single_item_retention_reports_componentwise_debt_reduction() {
        let retained = run(1, vec![10], 2, ExposureOutcome {
            wrong_belief_us: 2,
            abstention_us: 3,
            missing_output_us: 1,
        });
        let evicted = run(2, vec![], 0, ExposureOutcome {
            wrong_belief_us: 10,
            abstention_us: 5,
            missing_output_us: 4,
        });
        let effect = evaluate_paired_intervention(&registry(), &retained, &evicted, &[10], 123).unwrap();
        assert_eq!(effect.delta_evict_minus_retain, ExposureDelta {
            wrong_belief_us: 8,
            abstention_us: 2,
            missing_output_us: 3,
        });
    }

    #[test]
    fn paired_intervention_rejects_different_future_stream() {
        let retained = run(1, vec![10], 2, ExposureOutcome::default());
        let mut evicted = run(2, vec![], 0, ExposureOutcome::default());
        evicted.context.future_stream_sha256 = [9; 32];
        assert_eq!(
            evaluate_paired_intervention(&registry(), &retained, &evicted, &[10]),
            Err(CounterfactualError::ContextMismatch)
        );
    }

    #[test]
    fn paired_intervention_rejects_other_selection_differences() {
        let retained = run(1, vec![10, 20], 5, ExposureOutcome::default());
        let evicted = run(2, vec![], 0, ExposureOutcome::default());
        assert_eq!(
            evaluate_paired_intervention(&registry(), &retained, &evicted, &[10]),
            Err(CounterfactualError::TreatmentDiffersByUnexpectedItems)
        );
    }

    #[test]
    fn treatment_order_is_seeded_reproducible_and_evaluator_side() {
        let first = randomize_treatment_order(77, 42, &[10, 20]).unwrap();
        let repeated = randomize_treatment_order(77, 42, &[10, 20]).unwrap();
        assert_eq!(first, repeated);
        assert_ne!(
            randomize_treatment_order(77, 42, &[10, 20]).unwrap(),
            randomize_treatment_order(78, 42, &[10, 20]).unwrap()
        );
        assert_eq!(
            randomize_treatment_order(77, 42, &[]),
            Err(CounterfactualError::EmptyIntervention)
        );
    }

    #[test]
    fn exact_subset_oracle_reports_separate_component_optima_and_regret() {
        let items = registry();
        let runs = vec![
            run(0, vec![], 0, ExposureOutcome { wrong_belief_us: 10, abstention_us: 8, missing_output_us: 6 }),
            run(1, vec![10], 2, ExposureOutcome { wrong_belief_us: 4, abstention_us: 8, missing_output_us: 5 }),
            run(2, vec![20], 3, ExposureOutcome { wrong_belief_us: 7, abstention_us: 2, missing_output_us: 4 }),
            run(3, vec![10, 20], 5, ExposureOutcome { wrong_belief_us: 3, abstention_us: 3, missing_output_us: 1 }),
        ];
        let result = qualify_exact_subsets(&items, &runs, &[10]).unwrap();
        assert_eq!(result.evaluated_subset_count, 4);
        assert_eq!(result.best_wrong_belief.selected_item_ids, vec![10, 20]);
        assert_eq!(result.best_abstention.selected_item_ids, vec![20]);
        assert_eq!(result.best_missing_output.selected_item_ids, vec![10, 20]);
        assert_eq!(result.candidate_regret, SubsetRegret {
            wrong_belief_us: 1,
            abstention_us: 6,
            missing_output_us: 4,
        });
    }

    #[test]
    fn exact_subset_oracle_rejects_incomplete_enumeration() {
        let runs = vec![run(0, vec![], 0, ExposureOutcome::default())];
        assert_eq!(
            qualify_exact_subsets(&registry(), &runs, &[]),
            Err(CounterfactualError::IncompleteSubsetEnumeration { expected: 4, actual: 1 })
        );
    }

    #[test]
    fn pareto_frontier_keeps_tradeoffs_and_removes_dominated_subsets() {
        let items = registry();
        let runs = vec![
            run(0, vec![], 0, ExposureOutcome { wrong_belief_us: 8, abstention_us: 8, missing_output_us: 8 }),
            run(1, vec![10], 2, ExposureOutcome { wrong_belief_us: 2, abstention_us: 7, missing_output_us: 7 }),
            run(2, vec![20], 3, ExposureOutcome { wrong_belief_us: 7, abstention_us: 2, missing_output_us: 7 }),
            run(3, vec![10, 20], 5, ExposureOutcome { wrong_belief_us: 1, abstention_us: 1, missing_output_us: 1 }),
        ];
        let result = qualify_exact_subsets(&items, &runs, &[10]).unwrap();
        assert_eq!(result.pareto_frontier.len(), 1);
        assert_eq!(result.pareto_frontier[0].selected_item_ids, vec![10, 20]);
    }

    #[test]
    fn group_intervention_captures_nonadditive_memory_interaction() {
        let both_retained = run(1, vec![10, 20], 5, ExposureOutcome {
            wrong_belief_us: 2,
            abstention_us: 0,
            missing_output_us: 0,
        });
        let neither_retained = run(2, vec![], 0, ExposureOutcome {
            wrong_belief_us: 7,
            abstention_us: 0,
            missing_output_us: 0,
        });
        let only_first_retained = run(3, vec![10], 2, ExposureOutcome {
            wrong_belief_us: 7,
            abstention_us: 0,
            missing_output_us: 0,
        });
        let only_second_retained = run(4, vec![20], 3, ExposureOutcome {
            wrong_belief_us: 7,
            abstention_us: 0,
            missing_output_us: 0,
        });
        let first_alone = evaluate_paired_intervention(
            &registry(), &only_first_retained, &neither_retained, &[10], 123
        ).unwrap();
        let second_alone = evaluate_paired_intervention(
            &registry(), &only_second_retained, &neither_retained, &[20], 123
        ).unwrap();
        let together = evaluate_paired_intervention(
            &registry(), &both_retained, &neither_retained, &[10, 20], 123
        ).unwrap();
        assert_eq!(first_alone.delta_evict_minus_retain.wrong_belief_us, 0);
        assert_eq!(second_alone.delta_evict_minus_retain.wrong_belief_us, 0);
        assert_eq!(together.delta_evict_minus_retain.wrong_belief_us, 5);
        assert_eq!(together.retained_byte_us, 500);
        assert_eq!(together.evicted_byte_us, 0);
    }

    #[test]
    fn over_budget_and_inconsistent_exposure_are_rejected() {
        let mut over_budget = run(1, vec![10, 20], 5, ExposureOutcome::default());
        over_budget.context.byte_budget = 4;
        assert!(matches!(
            validate_run(&registry(), &over_budget),
            Err(CounterfactualError::OutOfBudget { .. })
        ));
        let invalid = run(2, vec![], 0, ExposureOutcome {
            wrong_belief_us: 80,
            abstention_us: 20,
            missing_output_us: 1,
        });
        assert!(matches!(
            validate_run(&registry(), &invalid),
            Err(CounterfactualError::ExposureExceedsEpisode { .. })
        ));
    }

    #[test]
    fn actual_compute_refresh_observation_and_energy_budgets_are_enforced() {
        let mut invalid_compute = run(1, vec![], 0, ExposureOutcome::default());
        invalid_compute.measured_cost.compute_used = 1_001;
        assert!(matches!(
            validate_run(&registry(), &invalid_compute),
            Err(CounterfactualError::ComputeOutOfBudget { .. })
        ));

        let mut invalid_energy = run(2, vec![], 0, ExposureOutcome::default());
        invalid_energy.measured_cost.measured_energy_nj = Some(10_001);
        assert!(matches!(
            validate_run(&registry(), &invalid_energy),
            Err(CounterfactualError::EnergyOutOfBudget { .. })
        ));

        let mut missing_energy = run(3, vec![], 0, ExposureOutcome::default());
        missing_energy.measured_cost.measured_energy_nj = None;
        assert_eq!(
            validate_run(&registry(), &missing_energy),
            Err(CounterfactualError::EnergyMeasurementMissing)
        );
    }

    #[test]
    fn duplicate_item_and_selection_ids_fail_closed() {
        let duplicate_registry = vec![
            MemoryItem { item_id: 1, bytes: 1 },
            MemoryItem { item_id: 1, bytes: 2 },
        ];
        assert_eq!(
            validate_item_registry(&duplicate_registry),
            Err(CounterfactualError::DuplicateItemId { item_id: 1 })
        );
        let invalid_selection = run(1, vec![20, 10], 5, ExposureOutcome::default());
        assert_eq!(
            validate_run(&registry(), &invalid_selection),
            Err(CounterfactualError::UnsortedOrDuplicateSelection)
        );
    }
}
