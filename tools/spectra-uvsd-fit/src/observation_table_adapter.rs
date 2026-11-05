// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed adapter from synthetic observed categories to the UVSD fitter.
//!
//! This module reads only observed category counts and action metadata. It
//! never reads oracle evidence, true categories or lapse indicators. Fit cells
//! are separated by condition, exact stimulus level and response-scale ID.

use std::collections::BTreeMap;
use std::fmt;

use super::scenario_dgp::{ActionCategorySummary, SyntheticDataset};
use super::sequential::{OperationalActionSet, StimulusClass};
use super::{CategoryCount, CategoryTable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FitGroupKey {
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

/// One condition/stimulus/scale cell and its explicit fit-eligibility result.
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

/// Complete adapter receipt. Every action summary is accounted for by one
/// fit group; non-ready groups preserve missingness and class counts.
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
    SummaryCountMismatch,
    UnknownSummaryAction,
    DuplicateActionSummary,
    MissingActionSummary,
    SummaryClassMismatch,
    SummaryCategoryCountMismatch,
    InconsistentCategorySummary,
    CountOverflow,
}

impl fmt::Display for ObservationFitAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ActionSetDigestMismatch => "dataset action-set digest does not match the requested action set",
            Self::SummaryCountMismatch => "dataset must contain exactly one summary for every action",
            Self::UnknownSummaryAction => "dataset summary references an unknown action",
            Self::DuplicateActionSummary => "duplicate summary for an action ID",
            Self::MissingActionSummary => "an action has no dataset summary",
            Self::SummaryClassMismatch => "summary stimulus class differs from its frozen action",
            Self::SummaryCategoryCountMismatch => "summary category vector differs from its response-scale profile",
            Self::InconsistentCategorySummary => "summary class/category/missing/RT counts do not conserve requested trials",
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

/// Convert observed category summaries into per-cell UVSD fitter tables.
///
/// Grouping is strictly by condition ID, exact stimulus-level bits and
/// response-scale ID. Any missing category response makes that cell
/// non-fit-eligible: the current fitter does not model missingness. No
/// missing row is dropped and no missingness mechanism is assumed.
pub fn adapt_synthetic_dataset_for_uvsd(
    dataset: &SyntheticDataset,
    action_set: &OperationalActionSet,
) -> Result<ObservationFitAdapterReceipt, ObservationFitAdapterError> {
    if dataset.action_set_digest != action_set.digest() {
        return Err(ObservationFitAdapterError::ActionSetDigestMismatch);
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
    for summary in &dataset.summaries {
        if !actions_by_id.contains_key(&summary.action_id) {
            return Err(ObservationFitAdapterError::UnknownSummaryAction);
        }
        validate_summary_conservation(summary)?;
        if summaries_by_id.insert(summary.action_id, summary).is_some() {
            return Err(ObservationFitAdapterError::DuplicateActionSummary);
        }
    }

    let mut groups = BTreeMap::<FitGroupKey, GroupAccumulator>::new();
    for action in action_set.actions() {
        let summary = summaries_by_id
            .get(&action.action_id)
            .copied()
            .ok_or(ObservationFitAdapterError::MissingActionSummary)?;
        if summary.stimulus_class != action.stimulus_class {
            return Err(ObservationFitAdapterError::SummaryClassMismatch);
        }
        let scale = dataset
            .scenario
            .response_scales
            .iter()
            .find(|scale| scale.response_scale_id == action.response_scale_id)
            .ok_or(ObservationFitAdapterError::SummaryCategoryCountMismatch)?;
        let category_count = scale.latent_criteria.len() + 1;
        if summary.category_counts.len() != category_count {
            return Err(ObservationFitAdapterError::SummaryCategoryCountMismatch);
        }
        let key = FitGroupKey {
            condition_id: action.condition_id,
            stimulus_level_bits: action.stimulus_level.to_bits(),
            response_scale_id: action.response_scale_id,
        };
        let accumulator = groups
            .entry(key)
            .or_insert_with(|| GroupAccumulator::new(key, category_count));
        if accumulator.category_count != category_count {
            return Err(ObservationFitAdapterError::SummaryCategoryCountMismatch);
        }
        accumulator.source_action_ids.push(action.action_id);
        match action.stimulus_class {
            StimulusClass::Signal => {
                accumulator.requested_signal_trials = checked_add(
                    accumulator.requested_signal_trials,
                    summary.requested_trials,
                )?;
                accumulator.missing_signal_categories = checked_add(
                    accumulator.missing_signal_categories,
                    summary.missing_category,
                )?;
                add_counts(&mut accumulator.signal_counts, &summary.category_counts)?;
            }
            StimulusClass::Noise => {
                accumulator.requested_noise_trials = checked_add(
                    accumulator.requested_noise_trials,
                    summary.requested_trials,
                )?;
                accumulator.missing_noise_categories = checked_add(
                    accumulator.missing_noise_categories,
                    summary.missing_category,
                )?;
                add_counts(&mut accumulator.noise_counts, &summary.category_counts)?;
            }
        }
        accumulator.rt_observed =
            checked_add(accumulator.rt_observed, summary.rt_observed)?;
        accumulator.rt_timeouts =
            checked_add(accumulator.rt_timeouts, summary.rt_timeouts)?;
        accumulator.rt_out_of_range =
            checked_add(accumulator.rt_out_of_range, summary.rt_out_of_range)?;
        accumulator.rt_not_collected =
            checked_add(accumulator.rt_not_collected, summary.rt_not_collected)?;
    }

    let mut fit_groups = Vec::with_capacity(groups.len());
    for (_, mut group) in groups {
        group.source_action_ids.sort_unstable();
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
            match CategoryTable::new(categories) {
                Ok(table) => Some(table),
                Err(_) => None,
            }
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
    if rt_total != summary.requested_trials || summary.lapse_count_oracle > summary.requested_trials {
        return Err(ObservationFitAdapterError::InconsistentCategorySummary);
    }
    Ok(())
}

fn add_counts(target: &mut [u64], source: &[u64]) -> Result<(), ObservationFitAdapterError> {
    if target.len() != source.len() {
        return Err(ObservationFitAdapterError::SummaryCategoryCountMismatch);
    }
    for (target_count, source_count) in target.iter_mut().zip(source) {
        *target_count = checked_add(*target_count, *source_count)?;
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
        EvidenceFamily, ObserverScenario, ResponseScaleSpec, TrialRequest,
        generate_synthetic_dataset,
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
                action_id: if trial % 2 == 0 { 1 } else if trial % 4 == 1 { 2 } else { 3 },
                participant_id: 0,
                within_participant_trial: trial,
            })
            .collect()
    }

    #[test]
    fn same_level_signal_noise_actions_form_one_ready_table() {
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
        assert_eq!(receipt.groups.len(), 2);
        let ready = receipt.groups.iter()
            .find(|group| group.key.stimulus_level() == 1.0)
            .unwrap();
        assert_eq!(ready.disposition, FitDisposition::Ready);
        let table = ready.table.as_ref().unwrap();
        assert_eq!(table.signal_total(), 1_000);
        assert_eq!(table.noise_total(), 1_000);
        assert_eq!(ready.source_action_ids, vec![1, 2]);
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
        let group = receipt.groups.iter()
            .find(|group| group.key.stimulus_level() == 1.0)
            .unwrap();
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
