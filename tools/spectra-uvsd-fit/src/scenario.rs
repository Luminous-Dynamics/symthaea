// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, failure-preserving aggregation across frozen model scenarios.
//!
//! This module does not fit models or decide whether a model is adequate.
//! It aggregates independently computed per-action, per-scenario results into
//! the CandidateAssessment consumed by the sequential policy. The raw scenario
//! definitions and outcomes are retained in the returned batch receipt.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::sequential::{
    CandidateAssessment, OperationalActionSet,
};

/// A named data-generating / observation scenario and its normalized weight.
///
/// Weights describe the caller's declared scenario mixture for weighted
/// summaries. They do not determine the maximin or minimum-representativeness
/// operations; every scenario remains visible to those conservative summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioDefinition {
    pub scenario_id: u16,
    pub weight: f64,
}

/// Frozen, canonically ordered scenario bank.
///
/// The digest is a caller-provided commitment to canonical source data; this
/// type neither computes nor authenticates it.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioSet {
    scenarios: Vec<ScenarioDefinition>,
    digest: String,
}

impl ScenarioSet {
    pub fn new(
        mut scenarios: Vec<ScenarioDefinition>,
        digest: impl Into<String>,
    ) -> Result<Self, ScenarioAssessmentError> {
        if scenarios.is_empty() {
            return Err(ScenarioAssessmentError::EmptyScenarioSet);
        }
        let digest = digest.into();
        if digest.trim().is_empty() {
            return Err(ScenarioAssessmentError::MissingScenarioSetDigest);
        }

        for scenario in &scenarios {
            if !scenario.weight.is_finite() || scenario.weight <= 0.0 {
                return Err(ScenarioAssessmentError::InvalidScenarioWeight);
            }
        }
        scenarios.sort_by_key(|scenario| scenario.scenario_id);
        if scenarios
            .windows(2)
            .any(|pair| pair[0].scenario_id == pair[1].scenario_id)
        {
            return Err(ScenarioAssessmentError::DuplicateScenarioId);
        }

        let total_weight: f64 = scenarios.iter().map(|scenario| scenario.weight).sum();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(ScenarioAssessmentError::ScenarioWeightOverflow);
        }
        for scenario in &mut scenarios {
            scenario.weight /= total_weight;
        }

        Ok(Self { scenarios, digest })
    }

    pub fn scenarios(&self) -> &[ScenarioDefinition] {
        &self.scenarios
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn scenario(&self, scenario_id: u16) -> Option<&ScenarioDefinition> {
        self.scenarios
            .binary_search_by_key(&scenario_id, |scenario| scenario.scenario_id)
            .ok()
            .map(|index| &self.scenarios[index])
    }
}

/// Conservative objective used to collapse scenario-specific target utility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustUtility {
    /// Minimum target utility across every declared scenario.
    Maximin,
    /// Weighted mean utility in the lower tail containing tail_mass probability.
    ///
    /// Fractional mass at the tail boundary is integrated exactly; scenarios
    /// are never rounded to an integer count.
    LowerTailCvar { tail_mass: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioOutcome {
    pub action_id: u32,
    pub scenario_id: u16,
    pub operational_mapping_qualified: bool,
    pub model_adequacy_qualified: bool,
    pub target_information: f64,
    pub model_discrimination_information: f64,
    /// Scenario-specific representativeness score on [0, 1].
    pub representativeness: f64,
    /// Higher is better; negative values are allowed.
    pub target_utility: f64,
    pub participant_burden_seconds: f64,
}

/// Complete input/output receipt for a scenario aggregation campaign.
///
/// Outcomes are sorted by (action_id, scenario_id); assessments are sorted by
/// action_id. The digest is still caller supplied and must be formed from a
/// canonical serialization of this batch for use by PolicyReceipt.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioAssessmentBatch {
    pub scenario_set_digest: String,
    pub scenarios: Vec<ScenarioDefinition>,
    pub robust_utility: RobustUtility,
    pub outcomes: Vec<ScenarioOutcome>,
    pub assessments: Vec<CandidateAssessment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioAssessmentError {
    EmptyScenarioSet,
    MissingScenarioSetDigest,
    InvalidScenarioWeight,
    ScenarioWeightOverflow,
    DuplicateScenarioId,
    InvalidTailMass,
    OutcomeCountOverflow,
    IncompleteOutcomeCoverage,
    UnknownActionId,
    UnknownScenarioId,
    DuplicateActionScenarioOutcome,
    InvalidOutcomeValue,
}

impl fmt::Display for ScenarioAssessmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyScenarioSet => "scenario set must not be empty",
            Self::MissingScenarioSetDigest => "scenario-set digest must not be empty",
            Self::InvalidScenarioWeight => "scenario weights must be finite and positive",
            Self::ScenarioWeightOverflow => "scenario weight total overflowed or was nonpositive",
            Self::DuplicateScenarioId => "scenario IDs must be unique",
            Self::InvalidTailMass => "lower-tail CVaR mass must be finite and in (0, 1]",
            Self::OutcomeCountOverflow => "action by scenario outcome count overflowed usize",
            Self::IncompleteOutcomeCoverage => {
                "exactly one outcome is required for every action and scenario pair"
            }
            Self::UnknownActionId => "scenario outcome references an unknown action",
            Self::UnknownScenarioId => "scenario outcome references an unknown scenario",
            Self::DuplicateActionScenarioOutcome => {
                "duplicate outcome for an action and scenario pair"
            }
            Self::InvalidOutcomeValue => "scenario outcome contains an invalid numeric value",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ScenarioAssessmentError {}

/// Aggregate a complete action × scenario outcome matrix into policy inputs.
///
/// Weighted target information and model-discrimination information are
/// descriptive averages. Representativeness is the minimum across scenarios,
/// burden is the maximum, and model/mapping eligibility requires every
/// scenario to be qualified. Robust utility is either the minimum or the
/// exact weighted lower-tail CVaR. No scenario is dropped or reweighted after
/// an error or failed qualification.
pub fn aggregate_scenario_assessments(
    action_set: &OperationalActionSet,
    scenario_set: &ScenarioSet,
    outcomes: &[ScenarioOutcome],
    robust_utility: RobustUtility,
) -> Result<ScenarioAssessmentBatch, ScenarioAssessmentError> {
    validate_robust_utility(robust_utility)?;

    let expected_outcomes = action_set
        .actions()
        .len()
        .checked_mul(scenario_set.scenarios().len())
        .ok_or(ScenarioAssessmentError::OutcomeCountOverflow)?;
    if outcomes.len() != expected_outcomes {
        return Err(ScenarioAssessmentError::IncompleteOutcomeCoverage);
    }

    let mut outcome_by_pair = BTreeMap::new();
    for outcome in outcomes {
        if action_set
            .actions()
            .binary_search_by_key(&outcome.action_id, |action| action.action_id)
            .is_err()
        {
            return Err(ScenarioAssessmentError::UnknownActionId);
        }
        if scenario_set.scenario(outcome.scenario_id).is_none() {
            return Err(ScenarioAssessmentError::UnknownScenarioId);
        }
        validate_outcome(outcome)?;
        if outcome_by_pair
            .insert((outcome.action_id, outcome.scenario_id), outcome)
            .is_some()
        {
            return Err(ScenarioAssessmentError::DuplicateActionScenarioOutcome);
        }
    }

    if outcome_by_pair.len() != expected_outcomes {
        return Err(ScenarioAssessmentError::IncompleteOutcomeCoverage);
    }

    let mut canonical_outcomes = outcomes.to_vec();
    canonical_outcomes.sort_by_key(|outcome| (outcome.action_id, outcome.scenario_id));

    let mut assessments = Vec::with_capacity(action_set.actions().len());
    for action in action_set.actions() {
        let mut target_information = 0.0;
        let mut model_discrimination_information = 0.0;
        let mut representativeness = 1.0_f64;
        let mut participant_burden_seconds = 0.0_f64;
        let mut target_utility_by_scenario = Vec::with_capacity(scenario_set.scenarios().len());
        let mut operational_mapping_qualified = true;
        let mut model_adequacy_qualified = true;

        for scenario in scenario_set.scenarios() {
            let outcome = outcome_by_pair
                .get(&(action.action_id, scenario.scenario_id))
                .copied()
                .ok_or(ScenarioAssessmentError::IncompleteOutcomeCoverage)?;

            target_information += scenario.weight * outcome.target_information;
            model_discrimination_information +=
                scenario.weight * outcome.model_discrimination_information;
            representativeness = representativeness.min(outcome.representativeness);
            participant_burden_seconds =
                participant_burden_seconds.max(outcome.participant_burden_seconds);
            operational_mapping_qualified &=
                outcome.operational_mapping_qualified;
            model_adequacy_qualified &= outcome.model_adequacy_qualified;
            target_utility_by_scenario.push((
                scenario.scenario_id,
                scenario.weight,
                outcome.target_utility,
            ));
        }

        let worst_case_target_utility =
            aggregate_robust_utility(&target_utility_by_scenario, robust_utility);

        assessments.push(CandidateAssessment {
            action_id: action.action_id,
            operational_mapping_qualified,
            model_adequacy_qualified,
            target_information,
            model_discrimination_information,
            representativeness: Some(representativeness),
            worst_case_target_utility: Some(worst_case_target_utility),
            participant_burden_seconds: Some(participant_burden_seconds),
        });
    }

    Ok(ScenarioAssessmentBatch {
        scenario_set_digest: scenario_set.digest().to_owned(),
        scenarios: scenario_set.scenarios().to_vec(),
        robust_utility,
        outcomes: canonical_outcomes,
        assessments,
    })
}

fn validate_robust_utility(
    robust_utility: RobustUtility,
) -> Result<(), ScenarioAssessmentError> {
    if let RobustUtility::LowerTailCvar { tail_mass } = robust_utility {
        if !tail_mass.is_finite() || tail_mass <= 0.0 || tail_mass > 1.0 {
            return Err(ScenarioAssessmentError::InvalidTailMass);
        }
    }
    Ok(())
}

fn validate_outcome(outcome: &ScenarioOutcome) -> Result<(), ScenarioAssessmentError> {
    if !outcome.target_information.is_finite()
        || !outcome.model_discrimination_information.is_finite()
        || !outcome.representativeness.is_finite()
        || !outcome.target_utility.is_finite()
        || !outcome.participant_burden_seconds.is_finite()
        || outcome.target_information < 0.0
        || outcome.model_discrimination_information < 0.0
        || !(0.0..=1.0).contains(&outcome.representativeness)
        || outcome.participant_burden_seconds < 0.0
    {
        return Err(ScenarioAssessmentError::InvalidOutcomeValue);
    }
    Ok(())
}

fn aggregate_robust_utility(
    outcomes: &[(u16, f64, f64)],
    robust_utility: RobustUtility,
) -> f64 {
    match robust_utility {
        RobustUtility::Maximin => outcomes
            .iter()
            .map(|(_, _, utility)| *utility)
            .fold(f64::INFINITY, f64::min),
        RobustUtility::LowerTailCvar { tail_mass } => {
            let mut sorted = outcomes.to_vec();
            sorted.sort_by(|left, right| {
                left.2
                    .total_cmp(&right.2)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let mut remaining = tail_mass;
            let mut weighted_utility = 0.0;
            for (_, weight, utility) in sorted {
                if remaining <= 0.0 {
                    break;
                }
                let included_mass = weight.min(remaining);
                weighted_utility += included_mass * utility;
                remaining -= included_mass;
            }
            weighted_utility / tail_mass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools_placeholder;

    fn make_action_set() -> OperationalActionSet {
        OperationalActionSet::new(
            vec![super::super::sequential::OperationalAction {
                action_id: 7,
                audit_stratum: 1,
                condition_id: 3,
                stimulus_class: super::super::sequential::StimulusClass::Signal,
                stimulus_level: 0.25,
                response_scale_id: 2,
                rt_deadline_ms: Some(1_500),
            }],
            "sha256:action-set",
        )
        .unwrap()
    }

    fn make_scenario_set() -> ScenarioSet {
        ScenarioSet::new(
            vec![
                ScenarioDefinition {
                    scenario_id: 20,
                    weight: 0.3,
                },
                ScenarioDefinition {
                    scenario_id: 10,
                    weight: 0.7,
                },
            ],
            "sha256:scenario-set",
        )
        .unwrap()
    }

    fn outcome(
        action_id: u32,
        scenario_id: u16,
        information: f64,
        discrimination: f64,
        representativeness: f64,
        utility: f64,
        burden: f64,
    ) -> ScenarioOutcome {
        ScenarioOutcome {
            action_id,
            scenario_id,
            operational_mapping_qualified: true,
            model_adequacy_qualified: true,
            target_information: information,
            model_discrimination_information: discrimination,
            representativeness,
            target_utility: utility,
            participant_burden_seconds: burden,
        }
    }

    fn reference_outcomes() -> Vec<ScenarioOutcome> {
        vec![
            outcome(7, 10, 1.0, 2.0, 0.9, 10.0, 1.0),
            outcome(7, 20, 3.0, 4.0, 0.2, -5.0, 3.0),
        ]
    }

    #[test]
    fn scenario_set_sorts_and_normalizes_positive_weights() {
        let scenarios = make_scenario_set();
        assert_eq!(
            scenarios
                .scenarios()
                .iter()
                .map(|scenario| scenario.scenario_id)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert!((scenarios.scenarios()[0].weight - 0.7).abs() < 1.0e-12);
        assert!((scenarios.scenarios()[1].weight - 0.3).abs() < 1.0e-12);
    }

    #[test]
    fn aggregation_uses_weighted_information_minimum_representativeness_and_max_burden() {
        let batch = aggregate_scenario_assessments(
            &make_action_set(),
            &make_scenario_set(),
            &reference_outcomes(),
            RobustUtility::Maximin,
        )
        .unwrap();
        let assessment = &batch.assessments[0];
        assert!((assessment.target_information.unwrap() - 1.6).abs() < 1.0e-12);
        assert!((assessment.model_discrimination_information.unwrap() - 2.6).abs() < 1.0e-12);
        assert!((assessment.representativeness.unwrap() - 0.2).abs() < 1.0e-12);
        assert!((assessment.participant_burden_seconds.unwrap() - 3.0).abs() < 1.0e-12);
        assert!((assessment.worst_case_target_utility.unwrap() + 5.0).abs() < 1.0e-12);
        assert!(assessment.operational_mapping_qualified);
        assert!(assessment.model_adequacy_qualified);
    }

    #[test]
    fn lower_tail_cvar_integrates_fractional_scenario_mass() {
        let batch = aggregate_scenario_assessments(
            &make_action_set(),
            &make_scenario_set(),
            &reference_outcomes(),
            RobustUtility::LowerTailCvar { tail_mass: 0.5 },
        )
        .unwrap();
        // The worst 0.5 mass consists of all 0.3 mass at utility -5
        // and 0.2 mass from the next scenario at utility +10.
        assert!((batch.assessments[0].worst_case_target_utility.unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn outcome_input_order_does_not_change_batch_or_assessments() {
        let action_set = make_action_set();
        let scenario_set = make_scenario_set();
        let outcomes = reference_outcomes();
        let first = aggregate_scenario_assessments(
            &action_set,
            &scenario_set,
            &outcomes,
            RobustUtility::Maximin,
        )
        .unwrap();
        let mut reversed = outcomes;
        reversed.reverse();
        let second = aggregate_scenario_assessments(
            &action_set,
            &scenario_set,
            &reversed,
            RobustUtility::Maximin,
        )
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn failed_mapping_and_adequacy_remain_explicit_hard_gate_values() {
        let action_set = make_action_set();
        let scenario_set = make_scenario_set();
        let mut outcomes = reference_outcomes();
        outcomes[1].operational_mapping_qualified = false;
        outcomes[1].model_adequacy_qualified = false;
        let batch = aggregate_scenario_assessments(
            &action_set,
            &scenario_set,
            &outcomes,
            RobustUtility::Maximin,
        )
        .unwrap();
        assert!(!batch.assessments[0].operational_mapping_qualified);
        assert!(!batch.assessments[0].model_adequacy_qualified);
        assert_eq!(batch.outcomes.len(), 2);
    }

    #[test]
    fn missing_and_duplicate_action_scenario_pairs_fail_closed() {
        let action_set = make_action_set();
        let scenario_set = make_scenario_set();
        let mut missing = reference_outcomes();
        missing.pop();
        assert_eq!(
            aggregate_scenario_assessments(
                &action_set,
                &scenario_set,
                &missing,
                RobustUtility::Maximin,
            ),
            Err(ScenarioAssessmentError::IncompleteOutcomeCoverage)
        );

        let duplicate = vec![reference_outcomes()[0].clone(), reference_outcomes()[0].clone()];
        assert_eq!(
            aggregate_scenario_assessments(
                &action_set,
                &scenario_set,
                &duplicate,
                RobustUtility::Maximin,
            ),
            Err(ScenarioAssessmentError::DuplicateActionScenarioOutcome)
        );
    }

    #[test]
    fn unknown_ids_invalid_scores_and_invalid_tail_mass_are_rejected() {
        let action_set = make_action_set();
        let scenario_set = make_scenario_set();
        let mut unknown_action = reference_outcomes();
        unknown_action[0].action_id = 99;
        assert_eq!(
            aggregate_scenario_assessments(
                &action_set,
                &scenario_set,
                &unknown_action,
                RobustUtility::Maximin,
            ),
            Err(ScenarioAssessmentError::UnknownActionId)
        );

        let mut unknown_scenario = reference_outcomes();
        unknown_scenario[0].scenario_id = 99;
        assert_eq!(
            aggregate_scenario_assessments(
                &action_set,
                &scenario_set,
                &unknown_scenario,
                RobustUtility::Maximin,
            ),
            Err(ScenarioAssessmentError::UnknownScenarioId)
        );

        let mut invalid_score = reference_outcomes();
        invalid_score[0].representativeness = 1.1;
        assert_eq!(
            aggregate_scenario_assessments(
                &action_set,
                &scenario_set,
                &invalid_score,
                RobustUtility::Maximin,
            ),
            Err(ScenarioAssessmentError::InvalidOutcomeValue)
        );

        assert_eq!(
            aggregate_scenario_assessments(
                &action_set,
                &scenario_set,
                &reference_outcomes(),
                RobustUtility::LowerTailCvar { tail_mass: 0.0 },
            ),
            Err(ScenarioAssessmentError::InvalidTailMass)
        );
    }

    #[test]
    fn malformed_scenario_sets_fail_closed() {
        assert_eq!(
            ScenarioSet::new(Vec::new(), "sha256:scenarios"),
            Err(ScenarioAssessmentError::EmptyScenarioSet)
        );
        assert_eq!(
            ScenarioSet::new(
                vec![
                    ScenarioDefinition { scenario_id: 1, weight: 1.0 },
                    ScenarioDefinition { scenario_id: 1, weight: 2.0 },
                ],
                "sha256:scenarios",
            ),
            Err(ScenarioAssessmentError::DuplicateScenarioId)
        );
        assert_eq!(
            ScenarioSet::new(
                vec![ScenarioDefinition { scenario_id: 1, weight: f64::NAN }],
                "sha256:scenarios",
            ),
            Err(ScenarioAssessmentError::InvalidScenarioWeight)
        );
        assert_eq!(
            ScenarioSet::new(
                vec![
                    ScenarioDefinition { scenario_id: 1, weight: f64::MAX },
                    ScenarioDefinition { scenario_id: 2, weight: f64::MAX },
                ],
                "sha256:scenarios",
            ),
            Err(ScenarioAssessmentError::ScenarioWeightOverflow)
        );
    }

    #[test]
    fn batch_receipt_preserves_normalized_scenarios_and_canonical_outcomes() {
        let batch = aggregate_scenario_assessments(
            &make_action_set(),
            &make_scenario_set(),
            &reference_outcomes(),
            RobustUtility::LowerTailCvar { tail_mass: 0.5 },
        )
        .unwrap();
        assert_eq!(batch.scenario_set_digest, "sha256:scenario-set");
        assert_eq!(batch.scenarios.len(), 2);
        assert_eq!(batch.outcomes[0].scenario_id, 10);
        assert_eq!(batch.outcomes[1].scenario_id, 20);
        assert_eq!(batch.assessments[0].action_id, 7);
    }
}
