//! Evaluator-only exact higher-order interaction decomposition.
//!
//! This module computes the Möbius (Harsanyi-dividend) decomposition of a
//! complete binary action lattice. It isolates pure k-way interaction terms:
//! first-order action effects, pairwise synergy/antagonism, and higher-order
//! effects that cannot be explained by lower-order terms.
//!
//! It must remain evaluator-side. The decomposition is exact for the finite,
//! fully enumerated action set and the declared episode context; it is not a
//! population-level causal estimate or a Shapley allocation.

use std::collections::{BTreeMap, BTreeSet};

use super::joint_budget::{
    action_registry_sha256, JointActionDefinition, JointBudgetContext,
    JointBudgetError, JointExposure, JointSubsetRun, MAX_EXACT_JOINT_ACTIONS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExposureDividend {
    /// Positive = synergistic exposure reduction; negative = antagonism.
    pub wrong_belief_us: i128,
    pub abstention_us: i128,
    pub missing_output_us: i128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HigherOrderInteractionTerm {
    /// Bit mask in the evaluator's canonical sorted action registry.
    pub subset_mask: u64,
    /// Canonically sorted action IDs in this interaction term.
    pub action_ids: Vec<u64>,
    /// 1 = main effect, 2 = pairwise, 3+ = higher-order interaction.
    pub order: u32,
    /// Number of counterfactual subsets required for this term (2^order).
    pub required_counterfactual_subsets: usize,
    pub eligible_counterfactual_subsets: usize,
    pub excluded_incompatible_subsets: usize,
    pub excluded_over_budget_subsets: usize,
    /// Present only when every required subset is compatible and feasible.
    /// Missing coefficients are never replaced with zero.
    pub dividend: Option<ExposureDividend>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HigherOrderInteractionReport {
    pub episode_id: u64,
    pub task_sha256: [u8; 32],
    pub future_stream_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub action_registry_sha256: [u8; 32],
    pub evaluated_subset_count: usize,
    pub baseline_exposure: JointExposure,
    pub terms: Vec<HigherOrderInteractionTerm>,
}

/// Compute exact Möbius dividends over every non-empty action subset.
///
/// For exposure component E and subset T:
///
///   v(S) = E(empty) - E(S)
///   dividend(T) = sum_{S subseteq T} (-1)^(|T|-|S|) * v(S)
///
/// A positive dividend means that the joint action bundle reduces exposure
/// beyond lower-order terms; a negative dividend means antagonism. Every
/// coefficient is exact integer arithmetic over evaluator-measured outcomes.
///
/// A coefficient is emitted only if every required sub-subset is compatible
/// and within the episode's hard resource budget. Otherwise the coefficient is
/// None with explicit exclusion counts. No missing or infeasible intervention
/// is silently imputed as zero.
pub fn decompose_higher_order_interactions(
    registry: &[JointActionDefinition],
    runs: &[JointSubsetRun],
) -> Result<HigherOrderInteractionReport, JointBudgetError> {
    if registry.len() > MAX_EXACT_JOINT_ACTIONS {
        return Err(JointBudgetError::TooManyActions {
            requested: registry.len(),
            maximum: MAX_EXACT_JOINT_ACTIONS,
        });
    }

    let mut sorted_registry = registry.to_vec();
    sorted_registry.sort_unstable_by_key(|definition| definition.action_id);
    let ids: Vec<u64> = sorted_registry.iter().map(|entry| entry.action_id).collect();
    if ids.contains(&0) {
        return Err(JointBudgetError::ZeroActionId);
    }
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(JointBudgetError::DuplicateActionId {
            action_id: ids.windows(2).find(|pair| pair[0] == pair[1]).unwrap()[0],
        });
    }

    let registry_digest = action_registry_sha256(&sorted_registry)?;
    let groups: Vec<u64> = sorted_registry.iter().map(|entry| entry.conflict_group).collect();
    let item_index: BTreeMap<u64, usize> = ids
        .iter()
        .enumerate()
        .map(|(index, action_id)| (*action_id, index))
        .collect();

    let expected = 1_usize
        .checked_shl(u32::try_from(ids.len()).map_err(|_| JointBudgetError::ArithmeticOverflow)?)
        .ok_or(JointBudgetError::ArithmeticOverflow)?;
    if runs.len() != expected {
        return Err(JointBudgetError::IncompleteEnumeration {
            expected,
            actual: runs.len(),
        });
    }

    let mut by_mask = vec![None::<&JointSubsetRun>; expected];
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

        if run.selected_action_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(JointBudgetError::UnsortedOrDuplicateSelection);
        }
        let mut mask = 0_u64;
        for action_id in &run.selected_action_ids {
            let index = *item_index
                .get(action_id)
                .ok_or(JointBudgetError::UnknownActionId { action_id: *action_id })?;
            mask |= 1_u64 << index;
        }
        let slot = usize::try_from(mask).map_err(|_| JointBudgetError::ArithmeticOverflow)?;
        if by_mask[slot].replace(run).is_some() {
            return Err(JointBudgetError::DuplicateSubset { mask });
        }
    }

    if let Some(missing) = by_mask.iter().position(Option::is_none) {
        return Err(JointBudgetError::MissingSubset {
            mask: u64::try_from(missing).map_err(|_| JointBudgetError::ArithmeticOverflow)?,
        });
    }

    let context = context_reference.ok_or(JointBudgetError::IncompleteEnumeration {
        expected,
        actual: 0,
    })?;
    let runs_by_mask = by_mask
        .into_iter()
        .enumerate()
        .map(|(mask, run)| {
            run.ok_or(JointBudgetError::MissingSubset {
                mask: u64::try_from(mask).map_err(|_| JointBudgetError::ArithmeticOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let baseline = runs_by_mask[0].outcome;
    let mut terms = Vec::with_capacity(expected.saturating_sub(1));

    for mask in 1..expected {
        let mask_u64 = u64::try_from(mask).map_err(|_| JointBudgetError::ArithmeticOverflow)?;
        let action_ids = ids
            .iter()
            .enumerate()
            .filter_map(|(index, action_id)| {
                (mask_u64 & (1_u64 << index) != 0).then_some(*action_id)
            })
            .collect::<Vec<_>>();
        let order = u32::try_from(action_ids.len())
            .map_err(|_| JointBudgetError::ArithmeticOverflow)?;
        let required = 1_usize
            .checked_shl(order)
            .ok_or(JointBudgetError::ArithmeticOverflow)?;

        let mut eligible = 0_usize;
        let mut excluded_incompatible = 0_usize;
        let mut excluded_over_budget = 0_usize;
        let mut dividend = ExposureDividend {
            wrong_belief_us: 0,
            abstention_us: 0,
            missing_output_us: 0,
        };

        let mut subset = mask;
        loop {
            let subset_u64 =
                u64::try_from(subset).map_err(|_| JointBudgetError::ArithmeticOverflow)?;
            let is_compatible = compatible(subset_u64, &groups);
            let is_within_budget = runs_by_mask[subset]
                .measured_resources
                .fits_within(context.resource_budget);

            if !is_compatible {
                excluded_incompatible = excluded_incompatible
                    .checked_add(1)
                    .ok_or(JointBudgetError::ArithmeticOverflow)?;
            } else if !is_within_budget {
                excluded_over_budget = excluded_over_budget
                    .checked_add(1)
                    .ok_or(JointBudgetError::ArithmeticOverflow)?;
            } else {
                eligible = eligible
                    .checked_add(1)
                    .ok_or(JointBudgetError::ArithmeticOverflow)?;

                let outcome = runs_by_mask[subset].outcome;
                let sign_is_positive = ((order - subset.count_ones()) % 2) == 0;
                let sign = if sign_is_positive { 1_i128 } else { -1_i128 };
                dividend.wrong_belief_us = dividend.wrong_belief_us
                    .checked_add(sign * (i128::from(baseline.wrong_belief_us)
                        - i128::from(outcome.wrong_belief_us)))
                    .ok_or(JointBudgetError::ArithmeticOverflow)?;
                dividend.abstention_us = dividend.abstention_us
                    .checked_add(sign * (i128::from(baseline.abstention_us)
                        - i128::from(outcome.abstention_us)))
                    .ok_or(JointBudgetError::ArithmeticOverflow)?;
                dividend.missing_output_us = dividend.missing_output_us
                    .checked_add(sign * (i128::from(baseline.missing_output_us)
                        - i128::from(outcome.missing_output_us)))
                    .ok_or(JointBudgetError::ArithmeticOverflow)?;
            }

            if subset == 0 {
                break;
            }
            subset = (subset - 1) & mask;
        }

        let dividend = if eligible == required {
            Some(dividend)
        } else {
            None
        };

        terms.push(HigherOrderInteractionTerm {
            subset_mask: mask_u64,
            action_ids,
            order,
            required_counterfactual_subsets: required,
            eligible_counterfactual_subsets: eligible,
            excluded_incompatible_subsets: excluded_incompatible,
            excluded_over_budget_subsets: excluded_over_budget,
            dividend,
        });
    }

    Ok(HigherOrderInteractionReport {
        episode_id: context.episode_id,
        task_sha256: context.task_sha256,
        future_stream_sha256: context.future_stream_sha256,
        model_sha256: context.model_sha256,
        action_registry_sha256: registry_digest,
        evaluated_subset_count: expected,
        baseline_exposure: baseline,
        terms,
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
    use super::super::joint_budget::JointResourceUse;

    fn registry(groups: &[u64]) -> Vec<JointActionDefinition> {
        groups.iter().enumerate().map(|(index, group)| {
            let action_id = u64::try_from(index + 1).unwrap() * 10;
            let mut semantics = [0_u8; 32];
            semantics[..8].copy_from_slice(&action_id.to_le_bytes());
            JointActionDefinition {
                action_id,
                conflict_group: *group,
                action_semantics_sha256: semantics,
            }
        }).collect()
    }

    fn context(registry: &[JointActionDefinition]) -> JointBudgetContext {
        JointBudgetContext {
            episode_id: 99,
            task_sha256: [1; 32],
            future_stream_sha256: [2; 32],
            model_sha256: [3; 32],
            initial_state_sha256: [4; 32],
            candidate_feature_sha256: [5; 32],
            action_registry_sha256: action_registry_sha256(registry).unwrap(),
            total_exposure_us: 10_000,
            resource_budget: JointResourceUse {
                persistent_bytes: 1_000_000,
                retained_byte_us: 1_000_000,
                io_bytes: 1_000_000,
                latency_us: 1_000_000,
                compute_units: 1_000_000,
                observations: 1_000_000,
                energy_nj: 1_000_000,
                verification_units: 1_000_000,
            },
        }
    }

    fn full_table(
        registry: &[JointActionDefinition],
        outcome_for_mask: impl Fn(u64) -> [u64; 3],
    ) -> Vec<JointSubsetRun> {
        let count = 1_u64 << registry.len();
        let mut runs = Vec::new();
        for mask in 0..count {
            let selected_action_ids = registry.iter().enumerate()
                .filter_map(|(index, definition)| {
                    (mask & (1_u64 << index) != 0).then_some(definition.action_id)
                })
                .collect::<Vec<_>>();
            let outcome = outcome_for_mask(mask);
            runs.push(JointSubsetRun {
                context: context(registry),
                run_id: mask + 1,
                selected_action_ids,
                measured_resources: JointResourceUse::default(),
                outcome: JointExposure {
                    wrong_belief_us: outcome[0],
                    abstention_us: outcome[1],
                    missing_output_us: outcome[2],
                },
            });
        }
        runs
    }

    fn term<'a>(report: &'a HigherOrderInteractionReport, mask: u64) -> &'a HigherOrderInteractionTerm {
        report.terms.iter().find(|term| term.subset_mask == mask).unwrap()
    }

    #[test]
    fn additive_three_action_lattice_has_no_pair_or_three_way_interaction() {
        let reg = registry(&[0, 0, 0]);
        let runs = full_table(&reg, |mask| {
            let benefit = mask.count_ones() * 10;
            [1_000 - benefit as u64, 2_000 - benefit as u64, 3_000 - benefit as u64]
        });
        let report = decompose_higher_order_interactions(&reg, &runs).unwrap();
        for mask in 1_u64..8 {
            let coefficient = term(&report, mask).dividend.unwrap();
            if mask.count_ones() == 1 {
                assert_eq!(coefficient.wrong_belief_us, 10);
            } else {
                assert_eq!(coefficient, ExposureDividend {
                    wrong_belief_us: 0,
                    abstention_us: 0,
                    missing_output_us: 0,
                });
            }
        }
    }

    #[test]
    fn pure_three_way_synergy_is_not_misattributed_to_pairs() {
        let reg = registry(&[0, 0, 0]);
        let runs = full_table(&reg, |mask| {
            let singles_and_pairs = mask.count_ones() * 10;
            let higher_order = if mask == 0b111 { 50 } else { 0 };
            let benefit = (singles_and_pairs + higher_order) as u64;
            [1_000 - benefit, 1_500 - benefit, 2_000 - benefit]
        });
        let report = decompose_higher_order_interactions(&reg, &runs).unwrap();
        assert_eq!(term(&report, 0b011).dividend.unwrap().wrong_belief_us, 0);
        assert_eq!(term(&report, 0b101).dividend.unwrap().wrong_belief_us, 0);
        assert_eq!(term(&report, 0b110).dividend.unwrap().wrong_belief_us, 0);
        assert_eq!(term(&report, 0b111).dividend.unwrap().wrong_belief_us, 50);
    }

    #[test]
    fn pure_four_way_synergy_is_isolated_from_lower_order_terms() {
        let reg = registry(&[0, 0, 0, 0]);
        let runs = full_table(&reg, |mask| {
            let benefit = mask.count_ones() * 10 + if mask == 0b1111 { 40 } else { 0 };
            [2_000 - benefit as u64, 3_000 - benefit as u64, 4_000 - benefit as u64]
        });
        let report = decompose_higher_order_interactions(&reg, &runs).unwrap();
        for mask in 1_u64..0b1111 {
            if mask.count_ones() > 1 {
                assert_eq!(term(&report, mask).dividend.unwrap().wrong_belief_us, 0);
            }
        }
        assert_eq!(term(&report, 0b1111).dividend.unwrap().wrong_belief_us, 40);
    }

    #[test]
    fn antagonistic_three_way_interaction_is_negative() {
        let reg = registry(&[0, 0, 0]);
        let runs = full_table(&reg, |mask| {
            let benefit = mask.count_ones() as i64 * 10 - if mask == 0b111 { 35 } else { 0 };
            [
                u64::try_from(1_000_i64 - benefit).unwrap(),
                u64::try_from(1_500_i64 - benefit).unwrap(),
                u64::try_from(2_000_i64 - benefit).unwrap(),
            ]
        });
        let report = decompose_higher_order_interactions(&reg, &runs).unwrap();
        assert_eq!(term(&report, 0b111).dividend.unwrap().wrong_belief_us, -35);
    }

    #[test]
    fn infeasible_counterfactuals_are_reported_and_never_imputed_as_zero() {
        let reg = registry(&[0, 0]);
        let mut runs = full_table(&reg, |_| [1_000, 1_000, 1_000]);
        runs[3].measured_resources.observations = 1_000_001;
        let report = decompose_higher_order_interactions(&reg, &runs).unwrap();
        let pair = term(&report, 0b11);
        assert_eq!(pair.dividend, None);
        assert_eq!(pair.required_counterfactual_subsets, 4);
        assert_eq!(pair.eligible_counterfactual_subsets, 3);
        assert_eq!(pair.excluded_over_budget_subsets, 1);
    }

    #[test]
    fn mutually_exclusive_actions_cannot_receive_a_joint_interaction_coefficient() {
        let reg = registry(&[7, 7]);
        let runs = full_table(&reg, |_| [1_000, 1_000, 1_000]);
        let report = decompose_higher_order_interactions(&reg, &runs).unwrap();
        let pair = term(&report, 0b11);
        assert_eq!(pair.dividend, None);
        assert_eq!(pair.excluded_incompatible_subsets, 1);
        assert_eq!(pair.eligible_counterfactual_subsets, 3);
    }

    #[test]
    fn context_or_action_registry_mismatch_fails_closed() {
        let reg = registry(&[0, 0]);
        let mut runs = full_table(&reg, |_| [1_000, 1_000, 1_000]);
        runs[0].context.future_stream_sha256 = [99; 32];
        assert_eq!(
            decompose_higher_order_interactions(&reg, &runs),
            Err(JointBudgetError::ContextMismatch)
        );

        let mut runs = full_table(&reg, |_| [1_000, 1_000, 1_000]);
        runs[0].context.action_registry_sha256 = [99; 32];
        assert_eq!(
            decompose_higher_order_interactions(&reg, &runs),
            Err(JointBudgetError::ContextMismatch)
        );
    }

    #[test]
    fn incomplete_and_duplicate_subset_tables_fail_closed() {
        let reg = registry(&[0, 0]);
        let runs = full_table(&reg, |_| [1_000, 1_000, 1_000]);
        assert!(matches!(
            decompose_higher_order_interactions(&reg, &runs[..3]),
            Err(JointBudgetError::IncompleteEnumeration { .. })
        ));

        let mut duplicate = runs.clone();
        duplicate[3].selected_action_ids = duplicate[2].selected_action_ids.clone();
        assert!(matches!(
            decompose_higher_order_interactions(&reg, &duplicate),
            Err(JointBudgetError::DuplicateSubset { .. })
        ));
    }
}
