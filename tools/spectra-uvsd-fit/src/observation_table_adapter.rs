// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed adapter from synthetic observed trials to the UVSD fitter.
//!
//! The fit unit is participant × condition × stimulus level × response scale.
//! The adapter reads only observed trial fields and experimenter action
//! metadata. It does not read oracle evidence, true categories, lapse labels,
//! participant effects or latent criterion values.

use std::collections::BTreeMap;
use std::fmt;

use super::scenario_dgp::{
    ActionCategorySummary, RtStatus, SyntheticDataset, SyntheticTrial, TrialRequest,
};
use super::sequential::{OperationalActionSet, StimulusClass};
use super::{CategoryCount, CategoryTable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FitGroupKey {
    pub participant_id: u32,
    pub condition_id: u16,
    /// Exact IEEE-754 bit pattern from the frozen experimenter action.
    pub stimulus_level_bits: u64,
    pub response_scale_id: u16,
}

impl FitGroupKey {
    pub fn stimulus_level(self) -> f64 {
        f64::from_bits(self.stimulus_level_bits)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitDisposition {
    Ready,
    MissingCategoryResponses,
    MissingSignalClass,
    MissingNoiseClass,
    CategoryOrderOverflow,
    InvalidCategoryTable,
}

/// One participant/session-level fit cell and its explicit fit disposition.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationFitGroup {
    pub key: FitGroupKey,
    pub disposition: FitDisposition,
    pub table: Option<CategoryTable>,
    pub source_action_ids: Vec<u32>,
    pub requested_signal_trials: u64,
    pub requested_noise_trials: u64,
    pub observed_signal_trials: u64,
    pub observed_noise_trials: u64,
    pub missing_signal_categories: u64,
    pub missing_noise_categories: u64,
    pub rt_observed: u64,
    pub rt_timeouts: u64,
    pub rt_out_of_range: u64,
    pub rt_not_collected: u64,
}

/// Complete adapter receipt. Each participant/condition/stimulus/scale cell is
/// separate; no participant pooling is performed.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationFitAdapterReceipt {
    pub scenario_id: u16,
    pub scenario_digest: String,
    pub action_set_digest: String,
    pub groups: Vec<ObservationFitGroup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationFitAdapterError {
    ActionSetDigestMismatch,
    MissingScenarioDigest,
    SummaryCountMismatch,
    UnknownSummaryAction,
    DuplicateActionSummary,
    MissingActionSummary,
    SummaryClassMismatch,
    SummaryCategoryCountMismatch,
    InconsistentCategorySummary,
    DuplicateParticipantTrial,
    UnknownRequestedAction,
    UnrequestedTrial,
    DuplicateObservedTrial,
    RequestTrialActionMismatch,
    TrialActionMetadataMismatch,
    CategoryResponseOutOfRange,
    CountOverflow,
}

impl fmt::Display for ObservationFitAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ActionSetDigestMismatch => "dataset action-set digest does not match the requested action set",
            Self::MissingScenarioDigest => "dataset scenario digest must not be empty",
            Self::SummaryCountMismatch => "dataset must contain exactly one summary for every action",
            Self::UnknownSummaryAction => "dataset summary references an unknown action",
            Self::DuplicateActionSummary => "duplicate summary for an action ID",
            Self::MissingActionSummary => "an action has no dataset summary",
            Self::SummaryClassMismatch => "summary stimulus class differs from its frozen action",
            Self::SummaryCategoryCountMismatch => "summary category vector differs from its response-scale presentation",
            Self::InconsistentCategorySummary => "summary class/category/missing/RT counts do not conserve requested trials",
            Self::DuplicateParticipantTrial => "participant/trial identity is duplicated in the frozen request schedule",
            Self::UnknownRequestedAction => "trial request references an unknown experimenter action",
            Self::UnrequestedTrial => "observed trial has no matching request",
            Self::DuplicateObservedTrial => "more than one observation exists for a participant/trial identity",
            Self::RequestTrialActionMismatch => "observed trial action differs from the frozen request",
            Self::TrialActionMetadataMismatch => "observed action metadata differs from the frozen action definition",
            Self::CategoryResponseOutOfRange => "observed category is outside the presented response category vector",
            Self::CountOverflow => "category or trial count overflowed u64",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ObservationFitAdapterError {}

#[derive(Debug, Clone)]
struct GroupAccumulator {
    key: FitGroupKey,
    category_count: usize,
    source_action_ids: Vec<u32>,
    signal_counts: Vec<u64>,
    noise_counts: Vec<u64>,
    requested_signal_trials: u64,
    requested_noise_trials: u64,
    missing_signal_categories: u64,
    missing_noise_categories: u64,
    rt_observed: u64,
    rt_timeouts: u64,
    rt_out_of_range: u64,
    rt_not_collected: u64,
}

impl GroupAccumulator {
    fn new(key: FitGroupKey, category_count: usize) -> Self {
        Self {
            key,
            category_count,
            source_action_ids: Vec::new(),
            signal_counts: vec![0; category_count],
            noise_counts: vec![0; category_count],
            requested_signal_trials: 0,
            requested_noise_trials: 0,
            missing_signal_categories: 0,
            missing_noise_categories: 0,
            rt_observed: 0,
            rt_timeouts: 0,
            rt_out_of_range: 0,
            rt_not_collected: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct ActionObservedTotals {
    category_counts: Vec<u64>,
    requested_trials: u64,
    missing_category: u64,
    rt_observed: u64,
    rt_timeouts: u64,
    rt_out_of_range: u64,
    rt_not_collected: u64,
}

impl ActionObservedTotals {
    fn new(category_count: usize) -> Self {
        Self {
            category_counts: vec![0; category_count],
            requested_trials: 0,
            missing_category: 0,
            rt_observed: 0,
            rt_timeouts: 0,
            rt_out_of_range: 0,
            rt_not_collected: 0,
        }
    }
}

/// Convert a complete synthetic dataset into separate per-participant UVSD
/// fitter tables. Grouping is by participant, condition, exact stimulus level
/// and response-scale ID; signal/noise trials combine only inside that cell.
///
/// The adapter reads only `observed_category`, RT status, request identity and
/// frozen action metadata. It intentionally never accesses `oracle_*` fields.
pub fn adapt_synthetic_dataset_for_uvsd(
    dataset: &SyntheticDataset,
    action_set: &OperationalActionSet,
) -> Result<ObservationFitAdapterReceipt, ObservationFitAdapterError> {
    if dataset.action_set_digest != action_set.digest() {
        return Err(ObservationFitAdapterError::ActionSetDigestMismatch);
    }
    if dataset.scenario_digest.trim().is_empty() {
        return Err(ObservationFitAdapterError::MissingScenarioDigest);
    }
    if dataset.summaries.len() != action_set.actions().len() {
        return Err(ObservationFitAdapterError::SummaryCountMismatch);
    }

    let actions_by_id: BTreeMap<_, _> = action_set
        .actions()
        .iter()
        .map(|action| (action.action_id, action))
        .collect();
    let mut summaries_by_id = BTreeMap::new();
    let mut category_count_by_scale = BTreeMap::<u16, usize>::new();

    for summary in &dataset.summaries {
        let action = actions_by_id
            .get(&summary.action_id)
            .copied()
            .ok_or(ObservationFitAdapterError::UnknownSummaryAction)?;
        if summary.stimulus_class != action.stimulus_class {
            return Err(ObservationFitAdapterError::SummaryClassMismatch);
        }
        if summary.category_counts.len() < 3 {
            return Err(ObservationFitAdapterError::SummaryCategoryCountMismatch);
        }
        match category_count_by_scale.get(&action.response_scale_id) {
            Some(existing) if *existing != summary.category_counts.len() => {
                return Err(ObservationFitAdapterError::SummaryCategoryCountMismatch);
            }
            Some(_) => {}
            None => {
                category_count_by_scale.insert(
                    action.response_scale_id,
                    summary.category_counts.len(),
                );
            }
        }
        validate_summary_conservation(summary)?;
        if summaries_by_id.insert(summary.action_id, summary).is_some() {
            return Err(ObservationFitAdapterError::DuplicateActionSummary);
        }
    }
    for action in action_set.actions() {
        if !summaries_by_id.contains_key(&action.action_id) {
            return Err(ObservationFitAdapterError::MissingActionSummary);
        }
    }

    let mut request_by_identity = BTreeMap::new();
    for request in &dataset.requests {
        if !actions_by_id.contains_key(&request.action_id) {
            return Err(ObservationFitAdapterError::UnknownRequestedAction);
        }
        let identity = (request.participant_id, request.within_participant_trial);
        if request_by_identity.insert(identity, request.action_id).is_some() {
            return Err(ObservationFitAdapterError::DuplicateParticipantTrial);
        }
    }
    if dataset.requests.len() != dataset.trials.len() {
        return Err(ObservationFitAdapterError::SummaryCountMismatch);
    }

    let mut observed_trial_identities = std::collections::BTreeSet::new();
    let mut action_totals = BTreeMap::new();
    for action in action_set.actions() {
        let category_count = category_count_by_scale[&action.response_scale_id];
        action_totals.insert(action.action_id, ActionObservedTotals::new(category_count));
    }
    let mut groups = BTreeMap::<FitGroupKey, GroupAccumulator>::new();

    for trial in &dataset.trials {
        let identity = (trial.participant_id, trial.within_participant_trial);
        let requested_action_id = request_by_identity
            .get(&identity)
            .copied()
            .ok_or(ObservationFitAdapterError::UnrequestedTrial)?;
        if !observed_trial_identities.insert(identity) {
            return Err(ObservationFitAdapterError::DuplicateObservedTrial);
        }
        if requested_action_id != trial.action_id {
            return Err(ObservationFitAdapterError::RequestTrialActionMismatch);
        }
        let action = actions_by_id
            .get(&trial.action_id)
            .copied()
            .ok_or(ObservationFitAdapterError::UnknownRequestedAction)?;
        if trial.stimulus_class != action.stimulus_class
            || trial.stimulus_level.to_bits() != action.stimulus_level.to_bits()
            || trial.response_scale_id != action.response_scale_id
        {
            return Err(ObservationFitAdapterError::TrialActionMetadataMismatch);
        }
        let category_count = category_count_by_scale[&action.response_scale_id];
        if trial
            .observed_category
            .is_some_and(|category| category as usize >= category_count)
        {
            return Err(ObservationFitAdapterError::CategoryResponseOutOfRange);
        }

        let action_total = action_totals
            .get_mut(&action.action_id)
            .ok_or(ObservationFitAdapterError::UnknownRequestedAction)?;
        action_total.requested_trials = checked_add(action_total.requested_trials, 1)?;
        if let Some(category) = trial.observed_category {
            let count = action_total
                .category_counts
                .get_mut(category as usize)
                .ok_or(ObservationFitAdapterError::CategoryResponseOutOfRange)?;
            *count = checked_add(*count, 1)?;
        } else {
            action_total.missing_category = checked_add(action_total.missing_category, 1)?;
        }
        update_rt_totals(action_total, trial.rt_status)?;

        let key = FitGroupKey {
            participant_id: trial.participant_id,
            condition_id: action.condition_id,
            stimulus_level_bits: action.stimulus_level.to_bits(),
            response_scale_id: action.response_scale_id,
        };
        let group = groups
            .entry(key)
            .or_insert_with(|| GroupAccumulator::new(key, category_count));
        group.source_action_ids.push(action.action_id);
        match action.stimulus_class {
            StimulusClass::Signal => {
                group.requested_signal_trials = checked_add(group.requested_signal_trials, 1)?;
                if let Some(category) = trial.observed_category {
                    let count = group
                        .signal_counts
                        .get_mut(category as usize)
                        .ok_or(ObservationFitAdapterError::CategoryResponseOutOfRange)?;
                    *count = checked_add(*count, 1)?;
                } else {
                    group.missing_signal_categories =
                        checked_add(group.missing_signal_categories, 1)?;
                }
            }
            StimulusClass::Noise => {
                group.requested_noise_trials = checked_add(group.requested_noise_trials, 1)?;
                if let Some(category) = trial.observed_category {
                    let count = group
                        .noise_counts
                        .get_mut(category as usize)
                        .ok_or(ObservationFitAdapterError::CategoryResponseOutOfRange)?;
                    *count = checked_add(*count, 1)?;
                } else {
                    group.missing_noise_categories =
                        checked_add(group.missing_noise_categories, 1)?;
                }
            }
        }
        update_rt_group(group, trial.rt_status)?;
    }

    if observed_trial_identities.len() != request_by_identity.len() {
        return Err(ObservationFitAdapterError::SummaryCountMismatch);
    }
    for action in action_set.actions() {
        let summary = summaries_by_id[&action.action_id];
        let totals = &action_totals[&action.action_id];
        if totals.requested_trials != summary.requested_trials
            || totals.category_counts != summary.category_counts
            || totals.missing_category != summary.missing_category
            || totals.rt_observed != summary.rt_observed
            || totals.rt_timeouts != summary.rt_timeouts
            || totals.rt_out_of_range != summary.rt_out_of_range
            || totals.rt_not_collected != summary.rt_not_collected
        {
            return Err(ObservationFitAdapterError::InconsistentCategorySummary);
        }
    }

    let mut fit_groups = Vec::with_capacity(groups.len());
    for (_, mut group) in groups {
        group.source_action_ids.sort_unstable();
        group.source_action_ids.dedup();
        let observed_signal_trials = sum_counts(&group.signal_counts)?;
        let observed_noise_trials = sum_counts(&group.noise_counts)?;
        let disposition = if group.missing_signal_categories > 0
            || group.missing_noise_categories > 0
        {
            FitDisposition::MissingCategoryResponses
        } else if observed_signal_trials == 0 {
            FitDisposition::MissingSignalClass
        } else if observed_noise_trials == 0 {
            FitDisposition::MissingNoiseClass
        } else if group.category_count > i16::MAX as usize + 1 {
            FitDisposition::CategoryOrderOverflow
        } else {
            FitDisposition::Ready
        };

        let table = if disposition == FitDisposition::Ready {
            let mut categories = Vec::with_capacity(group.category_count);
            for index in 0..group.category_count {
                let order = i16::try_from(index)
                    .map_err(|_| ObservationFitAdapterError::SummaryCategoryCountMismatch)?;
                categories.push(CategoryCount {
                    order,
                    signal: group.signal_counts[index],
                    noise: group.noise_counts[index],
                });
            }
            CategoryTable::new(categories).ok()
        } else {
            None
        };
        let disposition = if disposition == FitDisposition::Ready && table.is_none() {
            FitDisposition::InvalidCategoryTable
        } else {
            disposition
        };

        fit_groups.push(ObservationFitGroup {
            key: group.key,
            disposition,
            table,
            source_action_ids: group.source_action_ids,
            requested_signal_trials: group.requested_signal_trials,
            requested_noise_trials: group.requested_noise_trials,
            observed_signal_trials,
            observed_noise_trials,
            missing_signal_categories: group.missing_signal_categories,
            missing_noise_categories: group.missing_noise_categories,
            rt_observed: group.rt_observed,
            rt_timeouts: group.rt_timeouts,
            rt_out_of_range: group.rt_out_of_range,
            rt_not_collected: group.rt_not_collected,
        });
    }

    Ok(ObservationFitAdapterReceipt {
        scenario_id: dataset.scenario.scenario_id,
        scenario_digest: dataset.scenario_digest.clone(),
        action_set_digest: dataset.action_set_digest.clone(),
        groups: fit_groups,
    })
}

fn validate_summary_conservation(
    summary: &ActionCategorySummary,
) -> Result<(), ObservationFitAdapterError> {
    let category_total = sum_counts(&summary.category_counts)?;
    let represented = checked_add(category_total, summary.missing_category)?;
    if represented != summary.requested_trials {
        return Err(ObservationFitAdapterError::InconsistentCategorySummary);
    }
    let rt_total = checked_add(summary.rt_observed, summary.rt_timeouts)?;
    let rt_total = checked_add(rt_total, summary.rt_out_of_range)?;
    let rt_total = checked_add(rt_total, summary.rt_not_collected)?;
    if rt_total != summary.requested_trials {
        return Err(ObservationFitAdapterError::InconsistentCategorySummary);
    }
    Ok(())
}

fn update_rt_totals(
    totals: &mut ActionObservedTotals,
    status: RtStatus,
) -> Result<(), ObservationFitAdapterError> {
    match status {
        RtStatus::NotCollected => {
            totals.rt_not_collected = checked_add(totals.rt_not_collected, 1)?
        }
        RtStatus::Observed => totals.rt_observed = checked_add(totals.rt_observed, 1)?,
        RtStatus::TimedOut => totals.rt_timeouts = checked_add(totals.rt_timeouts, 1)?,
        RtStatus::OutOfRange => {
            totals.rt_out_of_range = checked_add(totals.rt_out_of_range, 1)?
        }
    }
    Ok(())
}

fn update_rt_group(
    group: &mut GroupAccumulator,
    status: RtStatus,
) -> Result<(), ObservationFitAdapterError> {
    match status {
        RtStatus::NotCollected => {
            group.rt_not_collected = checked_add(group.rt_not_collected, 1)?
        }
        RtStatus::Observed => group.rt_observed = checked_add(group.rt_observed, 1)?,
        RtStatus::TimedOut => group.rt_timeouts = checked_add(group.rt_timeouts, 1)?,
        RtStatus::OutOfRange => {
            group.rt_out_of_range = checked_add(group.rt_out_of_range, 1)?
        }
    }
    Ok(())
}

fn sum_counts(values: &[u64]) -> Result<u64, ObservationFitAdapterError> {
    values
        .iter()
        .try_fold(0u64, |total, value| checked_add(total, *value))
}

fn checked_add(left: u64, right: u64) -> Result<u64, ObservationFitAdapterError> {
    left.checked_add(right)
        .ok_or(ObservationFitAdapterError::CountOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario_dgp::{
        EvidenceFamily, ObserverScenario, ResponseScaleSpec, SyntheticDataset, SyntheticTrial,
        TrialRequest, generate_synthetic_dataset,
    };
    use crate::sequential::{OperationalAction, OperationalActionSet, StimulusClass};

    fn actions() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 1,
                    audit_stratum: 1,
                    condition_id: 4,
                    stimulus_class: StimulusClass::Noise,
                    stimulus_level: 1.0,
                    response_scale_id: 5,
                    rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 2,
                    audit_stratum: 1,
                    condition_id: 4,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 1.0,
                    response_scale_id: 5,
                    rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 3,
                    audit_stratum: 2,
                    condition_id: 4,
                    stimulus_class: StimulusClass::Signal,
                    stimulus_level: 2.0,
                    response_scale_id: 5,
                    rt_deadline_ms: None,
                },
            ],
            "sha256:fit-action-set",
        )
        .unwrap()
    }

    fn scenario(criteria: Vec<f64>) -> ObserverScenario {
        ObserverScenario {
            scenario_id: 12,
            evidence_family: EvidenceFamily::Gaussian,
            base_signal_mean: 1.3,
            signal_mean_per_stimulus_unit: 0.3,
            signal_sigma: 1.2,
            criterion_offset: 0.0,
            criterion_drift_per_100_trials: 0.0,
            lapse_probability: 0.01,
            confidence_missing_probability: 0.0,
            confidence_missing_evidence_slope: 0.0,
            response_scales: vec![ResponseScaleSpec {
                response_scale_id: 5,
                latent_criteria: criteria,
            }],
            heterogeneity: None,
            rt_channel: None,
        }
    }

    fn requests(count: u32) -> Vec<TrialRequest> {
        (0..count)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 {
                    1
                } else if trial % 4 == 1 {
                    2
                } else {
                    3
                },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect()
    }

    fn synthetic_trial(
        action: &super::super::sequential::OperationalAction,
        participant_id: u32,
        trial_index: u32,
        observed_category: Option<u16>,
    ) -> SyntheticTrial {
        SyntheticTrial {
            action_id: action.action_id,
            participant_id,
            within_participant_trial: trial_index,
            stimulus_class: action.stimulus_class,
            stimulus_level: action.stimulus_level,
            response_scale_id: action.response_scale_id,
            oracle_evidence: 0.0,
            oracle_category: 0,
            oracle_lapse: false,
            oracle_signal_mean: 0.0,
            oracle_signal_sigma: 1.0,
            oracle_criterion_offset: 0.0,
            observed_category,
            rt_ms: None,
            rt_status: RtStatus::NotCollected,
        }
    }

    #[test]
    fn same_level_signal_noise_actions_form_one_table_per_participant() {
        let action_set = actions();
        let schedule = (0..2_000)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 { 1 } else { 2 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &action_set,
            &scenario(vec![-1.0, -0.2, 0.2, 1.0]),
            &schedule,
            "sha256:scenario",
            99,
        )
        .unwrap();
        let receipt = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        assert_eq!(receipt.groups.len(), 1);
        let ready = &receipt.groups[0];
        assert_eq!(ready.disposition, FitDisposition::Ready);
        let table = ready.table.as_ref().unwrap();
        assert_eq!(table.signal_total(), 1_000);
        assert_eq!(table.noise_total(), 1_000);
        assert_eq!(ready.source_action_ids, vec![1, 2]);
        assert_eq!(ready.key.participant_id, 0);
    }

    #[test]
    fn participants_are_never_pooled_into_one_uvsd_fit() {
        let action_set = actions();
        let schedule = (0..1_000)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 { 1 } else { 2 },
                participant_id: trial % 2,
                within_participant_trial: trial / 2,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &action_set,
            &scenario(vec![-1.0, 0.0, 1.0]),
            &schedule,
            "sha256:scenario",
            123,
        )
        .unwrap();
        let receipt = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        assert_eq!(receipt.groups.len(), 2);
        assert_ne!(receipt.groups[0].key.participant_id, receipt.groups[1].key.participant_id);
        for group in &receipt.groups {
            assert_eq!(group.disposition, FitDisposition::Ready);
            assert_eq!(group.table.as_ref().unwrap().signal_total(), 250);
            assert_eq!(group.table.as_ref().unwrap().noise_total(), 250);
        }
    }

    #[test]
    fn different_stimulus_levels_are_not_pooled() {
        let action_set = actions();
        let schedule = (0..900)
            .map(|trial| TrialRequest {
                action_id: if trial % 3 == 0 { 1 } else if trial % 3 == 1 { 2 } else { 3 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &action_set,
            &scenario(vec![-1.0, 0.0, 1.0]),
            &schedule,
            "sha256:scenario",
            12,
        )
        .unwrap();
        let receipt = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        assert_eq!(receipt.groups.len(), 2);
        let level_one = receipt.groups.iter()
            .find(|group| group.key.stimulus_level() == 1.0)
            .unwrap();
        let level_two = receipt.groups.iter()
            .find(|group| group.key.stimulus_level() == 2.0)
            .unwrap();
        assert_eq!(level_one.disposition, FitDisposition::Ready);
        assert_eq!(level_two.disposition, FitDisposition::MissingNoiseClass);
        assert_eq!(level_two.table, None);
    }

    #[test]
    fn missing_confidence_response_abstains_instead_of_dropping_rows() {
        let action_set = actions();
        let schedule = (0..1_000)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 { 1 } else { 2 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let mut spec = scenario(vec![-1.0, 0.0, 1.0]);
        spec.confidence_missing_probability = 0.2;
        let data = generate_synthetic_dataset(
            &action_set,
            &spec,
            &schedule,
            "sha256:scenario",
            44,
        )
        .unwrap();
        let receipt = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        let group = &receipt.groups[0];
        assert_eq!(group.disposition, FitDisposition::MissingCategoryResponses);
        assert!(group.missing_signal_categories + group.missing_noise_categories > 0);
        assert_eq!(group.table, None);
        assert_eq!(
            group.observed_signal_trials + group.missing_signal_categories,
            group.requested_signal_trials
        );
        assert_eq!(
            group.observed_noise_trials + group.missing_noise_categories,
            group.requested_noise_trials
        );
    }

    #[test]
    fn scale_ids_form_separate_fit_units() {
        let actions = OperationalActionSet::new(
            vec![
                OperationalAction {
                    action_id: 1, audit_stratum: 1, condition_id: 1,
                    stimulus_class: StimulusClass::Noise, stimulus_level: 1.0,
                    response_scale_id: 5, rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 2, audit_stratum: 1, condition_id: 1,
                    stimulus_class: StimulusClass::Signal, stimulus_level: 1.0,
                    response_scale_id: 5, rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 3, audit_stratum: 1, condition_id: 1,
                    stimulus_class: StimulusClass::Noise, stimulus_level: 1.0,
                    response_scale_id: 6, rt_deadline_ms: None,
                },
                OperationalAction {
                    action_id: 4, audit_stratum: 1, condition_id: 1,
                    stimulus_class: StimulusClass::Signal, stimulus_level: 1.0,
                    response_scale_id: 6, rt_deadline_ms: None,
                },
            ],
            "sha256:scale-action-set",
        ).unwrap();
        let mut spec = scenario(vec![-1.0, 0.0, 1.0]);
        spec.response_scales.push(ResponseScaleSpec {
            response_scale_id: 6,
            latent_criteria: vec![-1.0, -0.5, 0.5, 1.0],
        });
        let schedule = (0..800)
            .map(|trial| TrialRequest {
                action_id: (trial % 4) + 1,
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let data = generate_synthetic_dataset(
            &actions, &spec, &schedule, "sha256:scenario", 88,
        ).unwrap();
        let receipt = adapt_synthetic_dataset_for_uvsd(&data, &actions).unwrap();
        assert_eq!(receipt.groups.len(), 2);
        assert_ne!(receipt.groups[0].key.response_scale_id, receipt.groups[1].key.response_scale_id);
    }

    #[test]
    fn category_order_overflow_is_an_explicit_nonready_disposition() {
        let action_set = actions();
        let category_counts = vec![0u64; i16::MAX as usize + 2];
        let mut noise_counts = category_counts.clone();
        let mut signal_counts = category_counts.clone();
        noise_counts[0] = 1;
        signal_counts[0] = 1;
        let empty_counts = category_counts.clone();
        let make_summary = |action_id, stimulus_class, requested_trials, category_counts| {
            ActionCategorySummary {
                action_id,
                stimulus_class,
                requested_trials,
                category_counts,
                missing_category: 0,
                lapse_count_oracle: 0,
                rt_observed: 0,
                rt_timeouts: 0,
                rt_out_of_range: 0,
                rt_not_collected: requested_trials,
            }
        };
        let mut overflow_scenario = scenario(vec![-1.0, 0.0, 1.0]);
        overflow_scenario.response_scales[0].latent_criteria =
            (0..category_counts.len() - 1).map(|index| index as f64).collect();
        let mut trials = Vec::new();
        let mut requests = Vec::new();
        for (action_id, category) in [(1u32, 0u16), (2u32, 0u16)] {
            let action = action_set.actions().iter()
                .find(|action| action.action_id == action_id)
                .unwrap();
            requests.push(TrialRequest {
                action_id,
                participant_id: 0,
                within_participant_trial: action_id - 1,
            });
            trials.push(synthetic_trial(action, 0, action_id - 1, Some(category)));
        }
        let data = SyntheticDataset {
            scenario: overflow_scenario,
            scenario_digest: "sha256:scenario".to_owned(),
            action_set_digest: action_set.digest().to_owned(),
            root_seed: 1,
            prng_version: "splitmix64-v1",
            requests,
            trials,
            summaries: vec![
                make_summary(1, StimulusClass::Noise, 1, noise_counts),
                make_summary(2, StimulusClass::Signal, 1, signal_counts),
                make_summary(3, StimulusClass::Signal, 0, empty_counts),
            ],
        };
        let receipt = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        let group = receipt.groups.iter()
            .find(|group| group.key.stimulus_level() == 1.0)
            .unwrap();
        assert_eq!(group.disposition, FitDisposition::CategoryOrderOverflow);
        assert_eq!(group.table, None);
    }

    #[test]
    fn action_summary_permutation_does_not_change_adapter_receipt() {
        let action_set = actions();
        let schedule = (0..1_000)
            .map(|trial| TrialRequest {
                action_id: if trial % 2 == 0 { 1 } else { 2 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect::<Vec<_>>();
        let mut data = generate_synthetic_dataset(
            &action_set,
            &scenario(vec![-1.0, 0.0, 1.0]),
            &schedule,
            "sha256:scenario",
            7,
        )
        .unwrap();
        let first = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        data.summaries.reverse();
        let second = adapt_synthetic_dataset_for_uvsd(&data, &action_set).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn action_set_digest_mismatch_rejects() {
        let action_set = actions();
        let schedule = requests(100);
        let data = generate_synthetic_dataset(
            &action_set,
            &scenario(vec![-1.0, 0.0, 1.0]),
            &schedule,
            "sha256:scenario",
            7,
        )
        .unwrap();
        let wrong = OperationalActionSet::new(
            action_set.actions().to_vec(),
            "sha256:wrong",
        ).unwrap();
        assert_eq!(
            adapt_synthetic_dataset_for_uvsd(&data, &wrong),
            Err(ObservationFitAdapterError::ActionSetDigestMismatch)
        );
    }
}
