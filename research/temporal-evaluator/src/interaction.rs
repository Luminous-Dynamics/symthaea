//! Evaluator-only causal attribution for joint temporal action subsets.
//!
//! Reports exact paired marginal effects and pairwise difference-in-differences.
//! Positive interaction means an action pair reduces temporal exposure more
//! than the sum of their isolated effects in the same context; negative means
//! antagonism. Values are exact signed sums over explicitly counted eligible
//! contexts, not floating-point estimates.
//!
//! This module requires a complete subset table and identical context digests.
//! It excludes structurally incompatible or over-budget intervention contexts
//! and reports those exclusions. It must never be linked into the candidate.

use std::collections::{BTreeMap, BTreeSet};

use super::joint_budget::{
    action_registry_sha256, JointActionDefinition, JointBudgetContext, JointBudgetError,
    JointExposure, JointSubsetRun, MAX_EXACT_JOINT_ACTIONS,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactSignedMean {
    /// Sum of signed marginal effects; divide by samples for the exact mean.
    pub sum: i128,
    pub samples: u32,
}

impl ExactSignedMean {
    fn push(&mut self, value: i128) -> Result<(), JointBudgetError> {
        self.sum = self.sum.checked_add(value).ok_or(JointBudgetError::ArithmeticOverflow)?;
        self.samples = self.samples.checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExposureAttribution {
    /// Positive values mean reduced error exposure.
    pub wrong_belief_reduction: ExactSignedMean,
    pub abstention_reduction: ExactSignedMean,
    pub missing_output_reduction: ExactSignedMean,
}

impl ExposureAttribution {
    fn push_difference(&mut self, before: JointExposure, after: JointExposure) -> Result<(), JointBudgetError> {
        self.wrong_belief_reduction.push(i128::from(before.wrong_belief_us) - i128::from(after.wrong_belief_us))?;
        self.abstention_reduction.push(i128::from(before.abstention_us) - i128::from(after.abstention_us))?;
        self.missing_output_reduction.push(i128::from(before.missing_output_us) - i128::from(after.missing_output_us))?;
        Ok(())
    }

    fn push_interaction(
        &mut self,
        without_i: JointExposure,
        with_i: JointExposure,
        without_i_with_j: JointExposure,
        with_i_with_j: JointExposure,
    ) -> Result<(), JointBudgetError> {
        self.wrong_belief_reduction.push(
            i128::from(with_i.wrong_belief_us)
                + i128::from(without_i_with_j.wrong_belief_us)
                - i128::from(without_i.wrong_belief_us)
                - i128::from(with_i_with_j.wrong_belief_us)
        )?;
        self.abstention_reduction.push(
            i128::from(with_i.abstention_us)
                + i128::from(without_i_with_j.abstention_us)
                - i128::from(without_i.abstention_us)
                - i128::from(with_i_with_j.abstention_us)
        )?;
        self.missing_output_reduction.push(
            i128::from(with_i.missing_output_us)
                + i128::from(without_i_with_j.missing_output_us)
                - i128::from(without_i.missing_output_us)
                - i128::from(with_i_with_j.missing_output_us)
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionMarginalAttribution {
    pub action_id: u64,
    pub eligible_contexts: u32,
    pub excluded_incompatible_contexts: u32,
    pub excluded_over_budget_contexts: u32,
    pub exposure_reduction: ExposureAttribution,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairwiseInteractionAttribution {
    pub action_a: u64,
    pub action_b: u64,
    pub eligible_contexts: u32,
    pub excluded_incompatible_contexts: u32,
    pub excluded_over_budget_contexts: u32,
    /// Positive = synergy, negative = antagonism, zero = additive in the
    /// measured eligible contexts.
    pub interaction: ExposureAttribution,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointInteractionReport {
    pub task_sha256: [u8; 32],
    pub future_stream_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub action_registry_sha256: [u8; 32],
    pub evaluated_subset_count: usize,
    pub action_marginals: Vec<ActionMarginalAttribution>,
    pub pairwise_interactions: Vec<PairwiseInteractionAttribution>,
}

/// Calculate paired action effects and pairwise difference-in-differences.
///
/// For action i, the main effect is the mean of E(S)-E(S+i) over all contexts S
/// where both interventions are compatible and within the same hard budgets.
///
/// For pair i,j, interaction is:
///   E(S+i) + E(S+j) - E(S) - E(S+i+j)
/// A positive value means j increases i's reduction of exposure (synergy).
///
/// Only contexts where all required counterfactual subsets are both compatible
/// and budget-feasible contribute. Exclusions are reported instead of hidden.
pub fn analyze_joint_interactions(
    registry: &[JointActionDefinition],
    runs: &[JointSubsetRun],
) -> Result<JointInteractionReport, JointBudgetError> {
    if registry.len() > MAX_EXACT_JOINT_ACTIONS {
        return Err(JointBudgetError::TooManyActions {
            requested: registry.len(),
            maximum: MAX_EXACT_JOINT_ACTIONS,
        });
    }
    let mut sorted_registry = registry.to_vec();
    sorted_registry.sort_unstable_by_key(|definition| definition.action_id);
    if sorted_registry.iter().any(|definition| definition.action_id == 0) {
        return Err(JointBudgetError::ZeroActionId);
    }
    if let Some(pair) = sorted_registry.windows(2).find(|pair| pair[0].action_id == pair[1].action_id) {
        return Err(JointBudgetError::DuplicateActionId { action_id: pair[0].action_id });
    }
    let registry_digest = action_registry_sha256(&sorted_registry)?;
    let ids: Vec<u64> = sorted_registry.iter().map(|entry| entry.action_id).collect();
    let groups: Vec<u64> = sorted_registry.iter().map(|entry| entry.conflict_group).collect();
    let item_index: BTreeMap<u64, usize> = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();

    let expected = 1_usize
        .checked_shl(u32::try_from(ids.len()).map_err(|_| JointBudgetError::ArithmeticOverflow)?)
        .ok_or(JointBudgetError::ArithmeticOverflow)?;
    if runs.len() != expected {
        return Err(JointBudgetError::IncompleteEnumeration { expected, actual: runs.len() });
    }

    let mut by_mask: Vec<Option<&JointSubsetRun>> = vec![None; expected];
    let mut run_ids = BTreeSet::new();
    let mut context_reference: Option<&JointBudgetContext> = None;

    for run in runs {
        run.context.validate()?;
        if run.context.action_registry_sha256 != registry_digest {
            return Err(JointBudgetError::ContextMismatch);
        }
        if run.run_id == 0 || !run_ids.insert(run.run_id) {
            return Err(JointBudgetError::DuplicateRunId { run_id: run.run_id });
        }
        if let Some(reference) = context_reference {
            if reference != &run.context {
                return Err(JointBudgetError::ContextMismatch);
            }
        } else {
            context_reference = Some(&run.context);
        }
        run.outcome.validate(run.context.total_exposure_us)?;
        let mut mask = 0_u64;
        for action_id in &run.selected_action_ids {
            let index = *item_index.get(action_id)
                .ok_or(JointBudgetError::UnknownActionId { action_id: *action_id })?;
            mask |= 1_u64 << index;
        }
        if run.selected_action_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(JointBudgetError::UnsortedOrDuplicateSelection);
        }
        let slot = usize::try_from(mask).map_err(|_| JointBudgetError::ArithmeticOverflow)?;
        if by_mask[slot].replace(run).is_some() {
            return Err(JointBudgetError::DuplicateSubset { mask });
        }
    }
    if let Some(missing) = by_mask.iter().position(Option::is_none) {
        return Err(JointBudgetError::MissingSubset { mask: missing as u64 });
    }
    let context = context_reference.ok_or(JointBudgetError::IncompleteEnumeration {
        expected,
        actual: 0,
    })?;
    let by_mask: Vec<&JointSubsetRun> = by_mask
        .into_iter()
        .enumerate()
        .map(|(mask, run)| run.ok_or(JointBudgetError::MissingSubset { mask: mask as u64 }))
        .collect::<Result<_, _>>()?;
    let feasible = |mask: usize| -> bool {
        compatible(mask as u64, &groups)
            && by_mask[mask].measured_resources.fits_within(context.resource_budget)
    };

    let mut action_marginals = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let bit = 1_usize << i;
        let mut result = ActionMarginalAttribution {
            action_id: ids[i],
            eligible_contexts: 0,
            excluded_incompatible_contexts: 0,
            excluded_over_budget_contexts: 0,
            exposure_reduction: ExposureAttribution::default(),
        };
        for mask in 0..expected {
            if mask & bit != 0 {
                continue;
            }
            let with_action = mask | bit;
            if !feasible(mask) || !feasible(with_action) {
                if !compatible(mask as u64, &groups) || !compatible(with_action as u64, &groups) {
                    result.excluded_incompatible_contexts = result.excluded_incompatible_contexts
                        .checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
                } else {
                    result.excluded_over_budget_contexts = result.excluded_over_budget_contexts
                        .checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
                }
                continue;
            }
            let before = by_mask[mask].outcome;
            let after = by_mask[with_action].outcome;
            result.exposure_reduction.push_difference(before, after)?;
            result.eligible_contexts = result.eligible_contexts
                .checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
        }
        action_marginals.push(result);
    }

    let mut pairwise_interactions = Vec::new();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let bit_i = 1_usize << i;
            let bit_j = 1_usize << j;
            let mut result = PairwiseInteractionAttribution {
                action_a: ids[i],
                action_b: ids[j],
                eligible_contexts: 0,
                excluded_incompatible_contexts: 0,
                excluded_over_budget_contexts: 0,
                interaction: ExposureAttribution::default(),
            };
            for mask in 0..expected {
                if mask & bit_i != 0 || mask & bit_j != 0 {
                    continue;
                }
                let mask_i = mask | bit_i;
                let mask_j = mask | bit_j;
                let mask_ij = mask | bit_i | bit_j;
                let all_compatible = compatible(mask as u64, &groups)
                    && compatible(mask_i as u64, &groups)
                    && compatible(mask_j as u64, &groups)
                    && compatible(mask_ij as u64, &groups);
                if !all_compatible {
                    result.excluded_incompatible_contexts = result.excluded_incompatible_contexts
                        .checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
                    continue;
                }
                if !feasible(mask) || !feasible(mask_i) || !feasible(mask_j) || !feasible(mask_ij) {
                    result.excluded_over_budget_contexts = result.excluded_over_budget_contexts
                        .checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
                    continue;
                }
                result.interaction.push_interaction(
                    by_mask[mask].outcome,
                    by_mask[mask_i].outcome,
                    by_mask[mask_j].outcome,
                    by_mask[mask_ij].outcome,
                )?;
                result.eligible_contexts = result.eligible_contexts
                    .checked_add(1).ok_or(JointBudgetError::ArithmeticOverflow)?;
            }
            pairwise_interactions.push(result);
        }
    }

    Ok(JointInteractionReport {
        task_sha256: context.task_sha256,
        future_stream_sha256: context.future_stream_sha256,
        model_sha256: context.model_sha256,
        action_registry_sha256: registry_digest,
        evaluated_subset_count: expected,
        action_marginals,
        pairwise_interactions,
    })
}

fn compatible(mask: u64, groups: &[u64]) -> bool {
    let mut seen = BTreeSet::new();
    for (index, group) in groups.iter().enumerate() {
        if mask & (1_u64 << index) == 0 || *group == 0 {
            continue;
        }
        if !seen.insert(*group) {
            return false;
        }
    }
    true
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

    fn context(registry: &[JointActionDefinition]) -> JointBudgetContext {
        JointBudgetContext {
            episode_id: 77,
            task_sha256: [1; 32],
            future_stream_sha256: [2; 32],
            model_sha256: [3; 32],
            initial_state_sha256: [4; 32],
            candidate_feature_sha256: [5; 32],
            action_registry_sha256: action_registry_sha256(registry).unwrap(),
            total_exposure_us: 200,
            resource_budget: super::super::joint_budget::JointResourceUse {
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

    fn make_run(
        registry: &[JointActionDefinition],
        run_id: u64,
        selected: Vec<u64>,
        exposure: [u64; 3],
        observations: u64,
    ) -> JointSubsetRun {
        JointSubsetRun {
            context: context(registry),
            run_id,
            selected_action_ids: selected,
            measured_resources: super::super::joint_budget::JointResourceUse {
                persistent_bytes: 1,
                retained_byte_us: 1,
                io_bytes: 1,
                latency_us: 1,
                compute_units: 1,
                observations,
                energy_nj: 1,
                verification_units: 1,
            },
            outcome: JointExposure {
                wrong_belief_us: exposure[0],
                abstention_us: exposure[1],
                missing_output_us: exposure[2],
            },
        }
    }

    fn four_runs(reg: &[JointActionDefinition], values: [[u64; 3]; 4]) -> Vec<JointSubsetRun> {
        vec![
            make_run(reg, 1, vec![], values[0], 0),
            make_run(reg, 2, vec![10], values[1], 1),
            make_run(reg, 3, vec![20], values[2], 1),
            make_run(reg, 4, vec![10, 20], values[3], 2),
        ]
    }

    #[test]
    fn additive_actions_have_zero_pairwise_interaction() {
        let reg = registry(&[(10, 0), (20, 0)]);
        let runs = four_runs(&reg, [[100, 100, 100], [90, 90, 90], [80, 80, 80], [70, 70, 70]]);
        let report = analyze_joint_interactions(&reg, &runs).unwrap();
        let main_effect = &report.action_marginals[0];
        assert_eq!(main_effect.eligible_contexts, 2);
        assert_eq!(main_effect.exposure_reduction.wrong_belief_reduction.sum, 20);
        assert_eq!(main_effect.exposure_reduction.wrong_belief_reduction.samples, 2);
        let pair = &report.pairwise_interactions[0];
        assert_eq!(pair.eligible_contexts, 1);
        assert_eq!(pair.interaction.wrong_belief_reduction.sum, 0);
        assert_eq!(pair.interaction.abstention_reduction.sum, 0);
        assert_eq!(pair.interaction.missing_output_reduction.sum, 0);
    }

    #[test]
    fn complementary_actions_show_positive_synergy() {
        let reg = registry(&[(10, 0), (20, 0)]);
        let runs = four_runs(&reg, [[100, 100, 100], [90, 90, 90], [90, 90, 90], [60, 60, 60]]);
        let report = analyze_joint_interactions(&reg, &runs).unwrap();
        let pair = &report.pairwise_interactions[0];
        assert_eq!(pair.interaction.wrong_belief_reduction.sum, 20);
        assert_eq!(pair.interaction.wrong_belief_reduction.samples, 1);
    }

    #[test]
    fn redundant_actions_show_negative_antagonism() {
        let reg = registry(&[(10, 0), (20, 0)]);
        let runs = four_runs(&reg, [[100, 100, 100], [90, 90, 90], [90, 90, 90], [95, 95, 95]]);
        let report = analyze_joint_interactions(&reg, &runs).unwrap();
        assert_eq!(report.pairwise_interactions[0].interaction.wrong_belief_reduction.sum, -15);
    }

    #[test]
    fn mutually_exclusive_actions_have_no_eligible_pair_context() {
        let reg = registry(&[(10, 7), (20, 7)]);
        let runs = four_runs(&reg, [[100, 100, 100], [90, 90, 90], [90, 90, 90], [60, 60, 60]]);
        let report = analyze_joint_interactions(&reg, &runs).unwrap();
        let pair = &report.pairwise_interactions[0];
        assert_eq!(pair.eligible_contexts, 0);
        assert_eq!(pair.excluded_incompatible_contexts, 1);
    }

    #[test]
    fn over_budget_counterfactual_pairs_are_excluded_not_hidden() {
        let reg = registry(&[(10, 0), (20, 0)]);
        let mut runs = four_runs(&reg, [[100, 100, 100], [90, 90, 90], [90, 90, 90], [60, 60, 60]]);
        runs[3].measured_resources.observations = 101;
        let report = analyze_joint_interactions(&reg, &runs).unwrap();
        let pair = &report.pairwise_interactions[0];
        assert_eq!(pair.eligible_contexts, 0);
        assert_eq!(pair.excluded_over_budget_contexts, 1);
    }

    #[test]
    fn incomplete_enumeration_and_registry_mismatch_fail_closed() {
        let reg = registry(&[(10, 0), (20, 0)]);
        let mut runs = four_runs(&reg, [[100, 100, 100], [90, 90, 90], [90, 90, 90], [60, 60, 60]]);
        assert!(matches!(
            analyze_joint_interactions(&reg, &runs[..3]),
            Err(JointBudgetError::IncompleteEnumeration { .. })
        ));
        runs[0].context.action_registry_sha256 = [9; 32];
        assert_eq!(
            analyze_joint_interactions(&reg, &runs),
            Err(JointBudgetError::ContextMismatch)
        );
    }
}
