//! Exact, model-relative stochastic assay qualification for categorical outcomes.
//!
//! The implementation assumes independently exchangeable categorical replicate
//! outcomes under the null. It uses exact two-sample permutation tests with a
//! declared enumeration budget and Holm family-wise correction across every
//! planned non-equivalent-model-pair/assay comparison.
//!
//! It does not implement paired/clustered designs, continuous outcomes,
//! sequential stopping, general power analysis, or a universal equivalence test.
//! Missing, failed and censored observations are not treated as zero outcomes.
use crate::assay::{AssayId, CandidateModel, ModelId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStatus {
    Observed,
    Missing,
    Failed,
    Censored,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticAssayObservation {
    pub model_id: ModelId,
    pub assay_id: AssayId,
    pub trial_id: String,
    pub status: SampleStatus,
    pub outcome: Option<String>,
}

impl StochasticAssayObservation {
    pub fn observed(
        model_id: ModelId,
        assay_id: AssayId,
        trial_id: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            model_id,
            assay_id,
            trial_id: trial_id.into(),
            status: SampleStatus::Observed,
            outcome: Some(outcome.into()),
        }
    }

    pub fn unavailable(
        model_id: ModelId,
        assay_id: AssayId,
        trial_id: impl Into<String>,
        status: SampleStatus,
    ) -> Self {
        Self {
            model_id,
            assay_id,
            trial_id: trial_id.into(),
            status,
            outcome: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticAssayPlan {
    /// Family-wise significance level, strictly between zero and one.
    pub alpha: f64,
    /// Minimum practically relevant total-variation effect, in [0, 1].
    pub minimum_effect: f64,
    /// Minimum observed independent replicates required per model and assay.
    pub minimum_samples_per_model: usize,
    /// Maximum exact label allocations enumerated for any pair/assay test.
    pub maximum_permutations: usize,
}

impl Default for StochasticAssayPlan {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            minimum_effect: 0.10,
            minimum_samples_per_model: 3,
            maximum_permutations: 100_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticAssayFindingCode {
    EmptyCandidateSet,
    EmptyRegisteredAssaySet,
    DuplicateModelId,
    EmptyModelId,
    EmptyEquivalenceClass,
    DuplicateAssayId,
    EmptyAssayId,
    UnknownModelObservation,
    UnknownAssayObservation,
    EmptyTrialId,
    DuplicateTrialObservation,
    InvalidObservedOutcome,
    OutcomeOnUnavailableObservation,
    InvalidPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticAssayFinding {
    pub code: StochasticAssayFindingCode,
    pub model_id: Option<ModelId>,
    pub assay_id: Option<AssayId>,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticTestDisposition {
    /// Adjusted significance and minimum practical effect both pass.
    Distinguishable,
    /// Test ran, but the pre-registered evidence threshold was not met.
    /// This does not imply model equivalence.
    NotDistinguished,
    /// Too few observed independent samples to run the registered test.
    Underpowered,
    /// The exact permutation space exceeded the declared enumeration budget.
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StochasticPairComparison {
    pub first: ModelId,
    pub second: ModelId,
    pub assay_id: AssayId,
    pub first_sample_count: usize,
    pub second_sample_count: usize,
    pub first_missing_count: usize,
    pub second_missing_count: usize,
    pub first_failed_count: usize,
    pub second_failed_count: usize,
    pub first_censored_count: usize,
    pub second_censored_count: usize,
    pub total_variation_effect: Option<f64>,
    pub exact_p_value: Option<f64>,
    pub holm_adjusted_p_value: Option<f64>,
    pub disposition: StochasticTestDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticPairResolution {
    pub first: ModelId,
    pub second: ModelId,
    pub distinguishing_assays: Vec<AssayId>,
}

impl StochasticPairResolution {
    pub fn is_resolved(&self) -> bool {
        !self.distinguishing_assays.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StochasticAssayReport {
    pub findings: Vec<StochasticAssayFinding>,
    pub comparisons: Vec<StochasticPairComparison>,
    pub pair_resolutions: Vec<StochasticPairResolution>,
    pub non_equivalent_pair_count: usize,
    pub resolved_pair_count: usize,
    /// Number of registered pair × assay hypotheses in the Holm family.
    pub planned_comparison_count: usize,
}

impl StochasticAssayReport {
    /// A finite model-class-relative conservative-panel claim. It is applicable
    /// only when the declared class contains non-equivalent pairs.
    pub fn is_conservative_for_declared_classes(&self) -> Option<bool> {
        if self.non_equivalent_pair_count == 0 {
            return None;
        }
        Some(self.findings.is_empty() && self.resolved_pair_count == self.non_equivalent_pair_count)
    }

    pub fn resolved_pair_fraction(&self) -> Option<f64> {
        if self.non_equivalent_pair_count == 0 {
            return None;
        }
        Some(self.resolved_pair_count as f64 / self.non_equivalent_pair_count as f64)
    }
}

#[derive(Debug, Clone, Default)]
struct ObservationGroup {
    outcomes: Vec<String>,
    missing: usize,
    failed: usize,
    censored: usize,
}

fn push_finding(
    report: &mut StochasticAssayReport,
    code: StochasticAssayFindingCode,
    model_id: Option<ModelId>,
    assay_id: Option<AssayId>,
    detail: impl Into<String>,
) {
    report.findings.push(StochasticAssayFinding {
        code,
        model_id,
        assay_id,
        detail: detail.into(),
    });
}

fn valid_plan(plan: StochasticAssayPlan) -> bool {
    plan.alpha.is_finite()
        && plan.alpha > 0.0
        && plan.alpha < 1.0
        && plan.minimum_effect.is_finite()
        && (0.0..=1.0).contains(&plan.minimum_effect)
        && plan.minimum_samples_per_model >= 2
        && plan.maximum_permutations > 0
}

fn total_variation_from_counts(
    first_counts: &[usize],
    pooled_counts: &[usize],
    first_n: usize,
    second_n: usize,
) -> f64 {
    let mut distance = 0.0;
    for (first_count, pooled_count) in first_counts.iter().zip(pooled_counts.iter()) {
        let second_count = pooled_count - first_count;
        let first_frequency = *first_count as f64 / first_n as f64;
        let second_frequency = second_count as f64 / second_n as f64;
        distance += (first_frequency - second_frequency).abs();
    }
    0.5 * distance
}

fn combinations_within_budget(n: usize, k: usize, budget: usize) -> Option<usize> {
    let k = k.min(n.saturating_sub(k));
    let mut combinations = 1_u128;
    for index in 1..=k {
        let factor = (n - k + index) as u128;
        combinations = combinations.checked_mul(factor)? / index as u128;
        if combinations > budget as u128 {
            return None;
        }
    }
    usize::try_from(combinations).ok()
}

#[allow(clippy::too_many_arguments)]
fn enumerate_allocations(
    labels: &[usize],
    total_counts: &[usize],
    first_n: usize,
    second_n: usize,
    observed_effect: f64,
    start: usize,
    remaining: usize,
    selected_counts: &mut [usize],
    total_allocations: &mut usize,
    extreme_allocations: &mut usize,
) {
    if remaining == 0 {
        *total_allocations += 1;
        let effect = total_variation_from_counts(selected_counts, total_counts, first_n, second_n);
        if effect + 1e-12 >= observed_effect {
            *extreme_allocations += 1;
        }
        return;
    }

    let final_start = labels.len() - remaining;
    for index in start..=final_start {
        selected_counts[labels[index]] += 1;
        enumerate_allocations(
            labels,
            total_counts,
            first_n,
            second_n,
            observed_effect,
            index + 1,
            remaining - 1,
            selected_counts,
            total_allocations,
            extreme_allocations,
        );
        selected_counts[labels[index]] -= 1;
    }
}

/// Exact two-sample permutation test for categorical response distributions.
/// Returns (observed total-variation distance, exact p-value), or None when
/// the number of allocations exceeds the declared budget.
fn exact_permutation_tv(
    first: &[String],
    second: &[String],
    maximum_permutations: usize,
) -> Option<(f64, f64)> {
    let first_n = first.len();
    let second_n = second.len();
    let total_n = first_n + second_n;
    if first_n == 0 || second_n == 0 {
        return None;
    }
    combinations_within_budget(total_n, first_n, maximum_permutations)?;

    let mut category_ids = BTreeMap::<String, usize>::new();
    let mut pooled_counts = Vec::<usize>::new();
    let mut labels = Vec::<usize>::with_capacity(total_n);

    for outcome in first.iter().chain(second.iter()) {
        let category_id = match category_ids.get(outcome) {
            Some(id) => *id,
            None => {
                let id = category_ids.len();
                category_ids.insert(outcome.clone(), id);
                pooled_counts.push(0);
                id
            }
        };
        labels.push(category_id);
        pooled_counts[category_id] += 1;
    }

    let mut observed_first_counts = vec![0_usize; category_ids.len()];
    for label in labels.iter().take(first_n) {
        observed_first_counts[*label] += 1;
    }
    let observed_effect =
        total_variation_from_counts(&observed_first_counts, &pooled_counts, first_n, second_n);

    let mut selected_counts = vec![0_usize; category_ids.len()];
    let mut total_allocations = 0;
    let mut extreme_allocations = 0;
    enumerate_allocations(
        &labels,
        &pooled_counts,
        first_n,
        second_n,
        observed_effect,
        0,
        first_n,
        &mut selected_counts,
        &mut total_allocations,
        &mut extreme_allocations,
    );

    if total_allocations == 0 {
        return None;
    }
    Some((
        observed_effect,
        extreme_allocations as f64 / total_allocations as f64,
    ))
}

/// Apply Holm's step-down family-wise correction across every planned
/// pair-by-assay comparison. Underpowered or budget-inconclusive tests remain
/// in the family as p=1 placeholders, making the correction conservative.
fn apply_holm_correction(comparisons: &mut [StochasticPairComparison], plan: StochasticAssayPlan) {
    let family_size = comparisons.len();
    if family_size == 0 {
        return;
    }

    let mut order: Vec<usize> = (0..family_size).collect();
    order.sort_by(|left, right| {
        let left_p = comparisons[*left].exact_p_value.unwrap_or(1.0);
        let right_p = comparisons[*right].exact_p_value.unwrap_or(1.0);
        left_p.total_cmp(&right_p).then_with(|| left.cmp(right))
    });

    let mut running_adjusted = 0.0_f64;
    for (rank, index) in order.into_iter().enumerate() {
        let Some(p_value) = comparisons[index].exact_p_value else {
            continue;
        };
        let multiplier = (family_size - rank) as f64;
        let candidate_adjusted = (multiplier * p_value).min(1.0);
        running_adjusted = running_adjusted.max(candidate_adjusted);
        comparisons[index].holm_adjusted_p_value = Some(running_adjusted);

        let effect = comparisons[index].total_variation_effect.unwrap_or(0.0);
        comparisons[index].disposition =
            if running_adjusted <= plan.alpha && effect >= plan.minimum_effect {
                StochasticTestDisposition::Distinguishable
            } else {
                StochasticTestDisposition::NotDistinguished
            };
    }
}

/// Evaluate repeated categorical responses over a frozen finite model class.
///
/// Assumptions: replicates are independent and exchangeable under the null
/// within each candidate model/assay group. Matched-seed, paired, clustered,
/// continuous and sequential designs require separate qualified procedures.
/// All planned non-equivalent pair × assay hypotheses are included in one
/// Holm family; missing/underpowered tests remain unresolved.
pub fn evaluate_stochastic_assay_adequacy(
    candidates: &[CandidateModel],
    registered_assays: &[AssayId],
    observations: &[StochasticAssayObservation],
    plan: StochasticAssayPlan,
) -> StochasticAssayReport {
    let mut report = StochasticAssayReport::default();

    if candidates.is_empty() {
        push_finding(
            &mut report,
            StochasticAssayFindingCode::EmptyCandidateSet,
            None,
            None,
            "candidate model set is empty",
        );
        return report;
    }

    let mut candidate_by_id = BTreeMap::<ModelId, &CandidateModel>::new();
    for candidate in candidates {
        if candidate.id.0.trim().is_empty() {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::EmptyModelId,
                Some(candidate.id.clone()),
                None,
                "candidate model ID is empty",
            );
            continue;
        }
        if candidate.target_equivalence_class.trim().is_empty() {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::EmptyEquivalenceClass,
                Some(candidate.id.clone()),
                None,
                "target equivalence class is empty",
            );
            continue;
        }
        if candidate_by_id
            .insert(candidate.id.clone(), candidate)
            .is_some()
        {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::DuplicateModelId,
                Some(candidate.id.clone()),
                None,
                "candidate model ID is duplicated",
            );
        }
    }

    let mut assay_set = BTreeSet::new();
    for assay_id in registered_assays {
        if assay_id.0.trim().is_empty() {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::EmptyAssayId,
                None,
                Some(assay_id.clone()),
                "registered assay ID is empty",
            );
            continue;
        }
        if !assay_set.insert(assay_id.clone()) {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::DuplicateAssayId,
                None,
                Some(assay_id.clone()),
                "registered assay ID is duplicated",
            );
        }
    }

    if assay_set.is_empty() {
        push_finding(
            &mut report,
            StochasticAssayFindingCode::EmptyRegisteredAssaySet,
            None,
            None,
            "no valid assay protocols were registered",
        );
    }

    let valid_candidates: Vec<&CandidateModel> = candidate_by_id.values().copied().collect();
    let mut non_equivalent_pairs = Vec::<(ModelId, ModelId)>::new();
    for first_index in 0..valid_candidates.len() {
        for second_index in (first_index + 1)..valid_candidates.len() {
            let first = valid_candidates[first_index];
            let second = valid_candidates[second_index];
            if first.target_equivalence_class != second.target_equivalence_class {
                non_equivalent_pairs.push((first.id.clone(), second.id.clone()));
            }
        }
    }
    report.non_equivalent_pair_count = non_equivalent_pairs.len();
    report.planned_comparison_count = non_equivalent_pairs.len() * assay_set.len();

    if !valid_plan(plan) {
        push_finding(
            &mut report,
            StochasticAssayFindingCode::InvalidPlan,
            None,
            None,
            "plan requires 0 < alpha < 1, 0 <= minimum effect <= 1, at least two samples per model, and a positive permutation budget",
        );
        return report;
    }

    let mut groups = BTreeMap::<(ModelId, AssayId), ObservationGroup>::new();
    let mut seen_trials = BTreeSet::<(ModelId, AssayId, String)>::new();

    for observation in observations {
        if !candidate_by_id.contains_key(&observation.model_id) {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::UnknownModelObservation,
                Some(observation.model_id.clone()),
                Some(observation.assay_id.clone()),
                "observation references an unknown or invalid candidate model",
            );
            continue;
        }
        if !assay_set.contains(&observation.assay_id) {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::UnknownAssayObservation,
                Some(observation.model_id.clone()),
                Some(observation.assay_id.clone()),
                "observation references an unregistered assay",
            );
            continue;
        }
        if observation.trial_id.trim().is_empty() {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::EmptyTrialId,
                Some(observation.model_id.clone()),
                Some(observation.assay_id.clone()),
                "trial ID is empty",
            );
            continue;
        }

        let trial_key = (
            observation.model_id.clone(),
            observation.assay_id.clone(),
            observation.trial_id.clone(),
        );
        if !seen_trials.insert(trial_key) {
            push_finding(
                &mut report,
                StochasticAssayFindingCode::DuplicateTrialObservation,
                Some(observation.model_id.clone()),
                Some(observation.assay_id.clone()),
                "trial ID is duplicated within a model/assay group",
            );
            continue;
        }

        let group = groups
            .entry((observation.model_id.clone(), observation.assay_id.clone()))
            .or_default();
        match observation.status {
            SampleStatus::Observed => match &observation.outcome {
                Some(outcome) if !outcome.trim().is_empty() => {
                    group.outcomes.push(outcome.clone());
                }
                _ => {
                    push_finding(
                        &mut report,
                        StochasticAssayFindingCode::InvalidObservedOutcome,
                        Some(observation.model_id.clone()),
                        Some(observation.assay_id.clone()),
                        "observed trial must carry a non-empty categorical outcome",
                    );
                }
            },
            SampleStatus::Missing => {
                group.missing += 1;
                if observation.outcome.is_some() {
                    push_finding(
                        &mut report,
                        StochasticAssayFindingCode::OutcomeOnUnavailableObservation,
                        Some(observation.model_id.clone()),
                        Some(observation.assay_id.clone()),
                        "missing trial must not carry an outcome",
                    );
                }
            }
            SampleStatus::Failed => {
                group.failed += 1;
                if observation.outcome.is_some() {
                    push_finding(
                        &mut report,
                        StochasticAssayFindingCode::OutcomeOnUnavailableObservation,
                        Some(observation.model_id.clone()),
                        Some(observation.assay_id.clone()),
                        "failed trial must not carry an outcome",
                    );
                }
            }
            SampleStatus::Censored => {
                group.censored += 1;
                if observation.outcome.is_some() {
                    push_finding(
                        &mut report,
                        StochasticAssayFindingCode::OutcomeOnUnavailableObservation,
                        Some(observation.model_id.clone()),
                        Some(observation.assay_id.clone()),
                        "censored trial must not carry an outcome",
                    );
                }
            }
        }
    }

    for (first, second) in &non_equivalent_pairs {
        for assay_id in &assay_set {
            let first_group = groups
                .get(&(first.clone(), assay_id.clone()))
                .cloned()
                .unwrap_or_default();
            let second_group = groups
                .get(&(second.clone(), assay_id.clone()))
                .cloned()
                .unwrap_or_default();

            let first_n = first_group.outcomes.len();
            let second_n = second_group.outcomes.len();
            let mut comparison = StochasticPairComparison {
                first: first.clone(),
                second: second.clone(),
                assay_id: assay_id.clone(),
                first_sample_count: first_n,
                second_sample_count: second_n,
                first_missing_count: first_group.missing,
                second_missing_count: second_group.missing,
                first_failed_count: first_group.failed,
                second_failed_count: second_group.failed,
                first_censored_count: first_group.censored,
                second_censored_count: second_group.censored,
                total_variation_effect: None,
                exact_p_value: None,
                holm_adjusted_p_value: None,
                disposition: StochasticTestDisposition::Underpowered,
            };

            if first_n >= plan.minimum_samples_per_model
                && second_n >= plan.minimum_samples_per_model
            {
                match exact_permutation_tv(
                    &first_group.outcomes,
                    &second_group.outcomes,
                    plan.maximum_permutations,
                ) {
                    Some((effect, p_value)) => {
                        comparison.total_variation_effect = Some(effect);
                        comparison.exact_p_value = Some(p_value);
                        comparison.disposition = StochasticTestDisposition::NotDistinguished;
                    }
                    None => {
                        comparison.disposition = StochasticTestDisposition::Inconclusive;
                    }
                }
            }
            report.comparisons.push(comparison);
        }
    }

    apply_holm_correction(&mut report.comparisons, plan);

    for (first, second) in non_equivalent_pairs {
        let distinguishing_assays: Vec<_> = report
            .comparisons
            .iter()
            .filter(|comparison| {
                comparison.first == first
                    && comparison.second == second
                    && comparison.disposition == StochasticTestDisposition::Distinguishable
            })
            .map(|comparison| comparison.assay_id.clone())
            .collect();
        let pair = StochasticPairResolution {
            first,
            second,
            distinguishing_assays,
        };
        if pair.is_resolved() {
            report.resolved_pair_count += 1;
        }
        report.pair_resolutions.push(pair);
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, class: &str) -> CandidateModel {
        CandidateModel::new(id, class)
    }

    fn assay(id: &str) -> AssayId {
        id.into()
    }

    fn observed(
        model_id: &str,
        assay_id: &str,
        trial: usize,
        outcome: &str,
    ) -> StochasticAssayObservation {
        StochasticAssayObservation::observed(
            model_id.into(),
            assay(assay_id),
            format!("trial-{trial}"),
            outcome,
        )
    }

    fn plan() -> StochasticAssayPlan {
        StochasticAssayPlan {
            alpha: 0.05,
            minimum_effect: 0.5,
            minimum_samples_per_model: 2,
            maximum_permutations: 1_000,
        }
    }

    #[test]
    fn clearly_separated_replicates_are_distinguished() {
        let candidates = vec![model("a", "class-a"), model("b", "class-b")];
        let assays = vec![assay("remove-producer")];
        let mut observations = Vec::new();
        for trial in 0..4 {
            observations.push(observed("a", "remove-producer", trial, "degraded"));
            observations.push(observed("b", "remove-producer", trial, "healthy"));
        }

        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &assays, &observations, plan());
        assert_eq!(report.non_equivalent_pair_count, 1);
        assert_eq!(report.resolved_pair_count, 1);
        assert_eq!(report.comparisons[0].total_variation_effect, Some(1.0));
        assert_eq!(report.comparisons[0].exact_p_value, Some(2.0 / 70.0));
        assert_eq!(
            report.comparisons[0].disposition,
            StochasticTestDisposition::Distinguishable
        );
        assert_eq!(report.is_conservative_for_declared_classes(), Some(true));
    }

    #[test]
    fn identical_response_distributions_are_not_distinguished() {
        let candidates = vec![model("a", "class-a"), model("b", "class-b")];
        let assays = vec![assay("assay-1")];
        let mut observations = Vec::new();
        for trial in 0..4 {
            observations.push(observed(
                "a",
                "assay-1",
                trial,
                if trial < 2 { "x" } else { "y" },
            ));
            observations.push(observed(
                "b",
                "assay-1",
                trial,
                if trial < 2 { "x" } else { "y" },
            ));
        }

        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &assays, &observations, plan());
        assert_eq!(report.comparisons[0].total_variation_effect, Some(0.0));
        assert_eq!(report.comparisons[0].exact_p_value, Some(1.0));
        assert_eq!(
            report.comparisons[0].disposition,
            StochasticTestDisposition::NotDistinguished
        );
        assert_eq!(report.is_conservative_for_declared_classes(), Some(false));
    }

    #[test]
    fn missing_and_failed_trials_are_not_zero_outcomes() {
        let candidates = vec![model("a", "class-a"), model("b", "class-b")];
        let assays = vec![assay("assay-1")];
        let mut observations = vec![
            observed("a", "assay-1", 0, "x"),
            observed("a", "assay-1", 1, "x"),
            observed("b", "assay-1", 0, "y"),
            observed("b", "assay-1", 1, "y"),
        ];
        observations.push(StochasticAssayObservation::unavailable(
            "a".into(),
            assay("assay-1"),
            "trial-2",
            SampleStatus::Missing,
        ));
        observations.push(StochasticAssayObservation::unavailable(
            "b".into(),
            assay("assay-1"),
            "trial-2",
            SampleStatus::Failed,
        ));

        let mut insufficient = plan();
        insufficient.minimum_samples_per_model = 3;
        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &assays, &observations, insufficient);
        assert_eq!(report.comparisons[0].first_missing_count, 1);
        assert_eq!(report.comparisons[0].second_failed_count, 1);
        assert_eq!(
            report.comparisons[0].disposition,
            StochasticTestDisposition::Underpowered
        );
        assert_eq!(report.comparisons[0].exact_p_value, None);
    }

    #[test]
    fn permutation_budget_exhaustion_is_inconclusive() {
        let candidates = vec![model("a", "class-a"), model("b", "class-b")];
        let assays = vec![assay("assay-1")];
        let observations = vec![
            observed("a", "assay-1", 0, "x"),
            observed("a", "assay-1", 1, "x"),
            observed("b", "assay-1", 0, "y"),
            observed("b", "assay-1", 1, "y"),
        ];
        let mut constrained = plan();
        constrained.maximum_permutations = 1;
        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &assays, &observations, constrained);
        assert_eq!(
            report.comparisons[0].disposition,
            StochasticTestDisposition::Inconclusive
        );
        assert_eq!(report.comparisons[0].exact_p_value, None);
        assert_eq!(report.is_conservative_for_declared_classes(), Some(false));
    }

    #[test]
    fn holm_correction_uses_the_full_planned_comparison_family() {
        let candidates = vec![
            model("a1", "class-a"),
            model("a2", "class-a"),
            model("b", "class-b"),
        ];
        let assays = vec![assay("assay-1")];
        let mut observations = Vec::new();
        for trial in 0..4 {
            observations.push(observed("a1", "assay-1", trial, "degraded"));
            observations.push(observed("a2", "assay-1", trial, "degraded"));
            observations.push(observed("b", "assay-1", trial, "healthy"));
        }

        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &assays, &observations, plan());
        assert_eq!(report.planned_comparison_count, 2);
        assert_eq!(report.comparisons.len(), 2);
        for comparison in &report.comparisons {
            assert!(comparison.holm_adjusted_p_value.unwrap() > 0.05);
            assert_eq!(
                comparison.disposition,
                StochasticTestDisposition::NotDistinguished
            );
        }
        assert_eq!(report.resolved_pair_count, 0);
    }

    #[test]
    fn target_equivalent_models_do_not_create_required_tests() {
        let candidates = vec![model("a", "same-class"), model("b", "same-class")];
        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &[assay("assay-1")], &[], plan());
        assert_eq!(report.non_equivalent_pair_count, 0);
        assert_eq!(report.comparisons.len(), 0);
        assert_eq!(report.is_conservative_for_declared_classes(), None);
    }

    #[test]
    fn duplicate_trial_observation_blocks_conservative_qualification() {
        let candidates = vec![model("a", "class-a"), model("b", "class-b")];
        let assays = vec![assay("assay-1")];
        let mut observations = Vec::new();
        for trial in 0..4 {
            observations.push(observed("a", "assay-1", trial, "x"));
            observations.push(observed("b", "assay-1", trial, "y"));
        }
        observations.push(observed("a", "assay-1", 0, "x"));

        let report =
            evaluate_stochastic_assay_adequacy(&candidates, &assays, &observations, plan());
        assert!(report.findings.iter().any(|finding| {
            finding.code == StochasticAssayFindingCode::DuplicateTrialObservation
        }));
        assert_eq!(report.is_conservative_for_declared_classes(), Some(false));
    }
}
