// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only causal-identification theorem for Economic Science.
//!
//! This test keeps four scientific objects separate:
//!
//! causal estimand != identification strategy != estimator != estimate
//!
//! Naming a strategy does not prove its assumptions. A strategy-bound estimate
//! remains a candidate causal estimate carrying its assumptions and lineage.

use symthaea_economics::{EconomicVariable, StateDomain, UnitId, VariableId};

#[derive(Debug, Clone, PartialEq, Eq)]
enum CausalError {
    EmptyText(&'static str),
    EmptyAssumptions,
    InvalidDesignContract,
    EstimandMismatch,
    StrategyMismatch,
    EstimatorMismatch,
    TreatmentMismatch,
    OutcomeMismatch,
    OutcomeUnitMismatch,
    AssociationalOutputNotCausal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EffectTarget {
    AveragePopulation,
    AverageTreated,
    Conditional { subgroup_id: String },
    Local { population_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CausalEstimand {
    id: String,
    treatment: VariableId,
    outcome: VariableId,
    outcome_unit: UnitId,
    population_id: String,
    intervention_contrast_id: String,
    horizon_id: String,
    target: EffectTarget,
}

impl CausalEstimand {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: impl Into<String>,
        treatment: VariableId,
        outcome: VariableId,
        outcome_unit: UnitId,
        population_id: impl Into<String>,
        intervention_contrast_id: impl Into<String>,
        horizon_id: impl Into<String>,
        target: EffectTarget,
    ) -> Result<Self, CausalError> {
        let id = id.into();
        let population_id = population_id.into();
        let intervention_contrast_id = intervention_contrast_id.into();
        let horizon_id = horizon_id.into();
        for (field, value) in [
            ("causal estimand id", id.as_str()),
            ("causal estimand population", population_id.as_str()),
            ("intervention contrast id", intervention_contrast_id.as_str()),
            ("causal estimand horizon", horizon_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CausalError::EmptyText(field));
            }
        }
        match &target {
            EffectTarget::Conditional { subgroup_id } if subgroup_id.trim().is_empty() => {
                return Err(CausalError::EmptyText("conditional estimand subgroup"));
            }
            EffectTarget::Local { population_id } if population_id.trim().is_empty() => {
                return Err(CausalError::EmptyText("local estimand population"));
            }
            _ => {}
        }
        Ok(Self {
            id,
            treatment,
            outcome,
            outcome_unit,
            population_id,
            intervention_contrast_id,
            horizon_id,
            target,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum IdentificationDesign {
    RandomizedAssignment {
        assignment_artifact_id: String,
        allocation_unit_id: String,
    },
    InstrumentalVariable {
        instrument: VariableId,
        first_stage_artifact_id: String,
        exclusion_assumption_id: String,
    },
    RegressionDiscontinuity {
        running_variable: VariableId,
        cutoff_atoms: i128,
        continuity_assumption_id: String,
    },
    DifferenceInDifferences {
        group_definition_id: String,
        intervention_time_id: String,
        parallel_trends_assumption_id: String,
    },
    SyntheticControl {
        donor_pool_id: String,
        preperiod_fit_artifact_id: String,
    },
    BackdoorAdjustment {
        adjustment_set: Vec<VariableId>,
        causal_graph_artifact_id: String,
    },
}

impl IdentificationDesign {
    fn validate(&self) -> Result<(), CausalError> {
        let nonempty = |value: &str| !value.trim().is_empty();
        let valid = match self {
            Self::RandomizedAssignment {
                assignment_artifact_id,
                allocation_unit_id,
            } => nonempty(assignment_artifact_id) && nonempty(allocation_unit_id),
            Self::InstrumentalVariable {
                first_stage_artifact_id,
                exclusion_assumption_id,
                ..
            } => nonempty(first_stage_artifact_id) && nonempty(exclusion_assumption_id),
            Self::RegressionDiscontinuity {
                continuity_assumption_id,
                ..
            } => nonempty(continuity_assumption_id),
            Self::DifferenceInDifferences {
                group_definition_id,
                intervention_time_id,
                parallel_trends_assumption_id,
            } => {
                nonempty(group_definition_id)
                    && nonempty(intervention_time_id)
                    && nonempty(parallel_trends_assumption_id)
            }
            Self::SyntheticControl {
                donor_pool_id,
                preperiod_fit_artifact_id,
            } => nonempty(donor_pool_id) && nonempty(preperiod_fit_artifact_id),
            Self::BackdoorAdjustment {
                adjustment_set,
                causal_graph_artifact_id,
            } => !adjustment_set.is_empty() && nonempty(causal_graph_artifact_id),
        };
        if valid {
            Ok(())
        } else {
            Err(CausalError::InvalidDesignContract)
        }
    }

    fn family_name(&self) -> &'static str {
        match self {
            Self::RandomizedAssignment { .. } => "randomized-assignment",
            Self::InstrumentalVariable { .. } => "instrumental-variable",
            Self::RegressionDiscontinuity { .. } => "regression-discontinuity",
            Self::DifferenceInDifferences { .. } => "difference-in-differences",
            Self::SyntheticControl { .. } => "synthetic-control",
            Self::BackdoorAdjustment { .. } => "backdoor-adjustment",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IdentificationStrategy {
    id: String,
    estimand_id: String,
    design: IdentificationDesign,
    assumptions: Vec<String>,
    diagnostic_artifact_ids: Vec<String>,
}

impl IdentificationStrategy {
    fn new(
        id: impl Into<String>,
        estimand: &CausalEstimand,
        design: IdentificationDesign,
        assumptions: Vec<String>,
        diagnostic_artifact_ids: Vec<String>,
    ) -> Result<Self, CausalError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(CausalError::EmptyText("identification strategy id"));
        }
        design.validate()?;
        if assumptions.is_empty() || assumptions.iter().any(|item| item.trim().is_empty()) {
            return Err(CausalError::EmptyAssumptions);
        }
        if diagnostic_artifact_ids
            .iter()
            .any(|item| item.trim().is_empty())
        {
            return Err(CausalError::EmptyText("identification diagnostic artifact"));
        }
        Ok(Self {
            id,
            estimand_id: estimand.id.clone(),
            design,
            assumptions,
            diagnostic_artifact_ids,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EstimatorSpecification {
    id: String,
    estimand_id: String,
    strategy_id: String,
    estimator_family_id: String,
    implementation_artifact_id: String,
    uncertainty_method_artifact_id: String,
}

impl EstimatorSpecification {
    fn new(
        id: impl Into<String>,
        estimand: &CausalEstimand,
        strategy: &IdentificationStrategy,
        estimator_family_id: impl Into<String>,
        implementation_artifact_id: impl Into<String>,
        uncertainty_method_artifact_id: impl Into<String>,
    ) -> Result<Self, CausalError> {
        if strategy.estimand_id != estimand.id {
            return Err(CausalError::EstimandMismatch);
        }
        let id = id.into();
        let estimator_family_id = estimator_family_id.into();
        let implementation_artifact_id = implementation_artifact_id.into();
        let uncertainty_method_artifact_id = uncertainty_method_artifact_id.into();
        for (field, value) in [
            ("estimator id", id.as_str()),
            ("estimator family id", estimator_family_id.as_str()),
            ("estimator implementation artifact", implementation_artifact_id.as_str()),
            ("estimator uncertainty method", uncertainty_method_artifact_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CausalError::EmptyText(field));
            }
        }
        Ok(Self {
            id,
            estimand_id: estimand.id.clone(),
            strategy_id: strategy.id.clone(),
            estimator_family_id,
            implementation_artifact_id,
            uncertainty_method_artifact_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EstimationOutputKind {
    Associational,
    StrategyBound,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EstimationOutput {
    kind: EstimationOutputKind,
    estimand_id: Option<String>,
    strategy_id: Option<String>,
    estimator_id: Option<String>,
    treatment: VariableId,
    outcome: VariableId,
    outcome_unit: UnitId,
    effect_atoms: i128,
    lower_atoms: i128,
    upper_atoms: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StrategyBoundCausalEstimateCandidate {
    estimand_id: String,
    strategy_id: String,
    estimator_id: String,
    design_family: &'static str,
    assumptions: Vec<String>,
    diagnostic_artifact_ids: Vec<String>,
    estimator_family_id: String,
    implementation_artifact_id: String,
    uncertainty_method_artifact_id: String,
    effect_atoms: i128,
    lower_atoms: i128,
    upper_atoms: i128,
}

fn qualify_causal_estimate(
    estimand: &CausalEstimand,
    strategy: &IdentificationStrategy,
    estimator: &EstimatorSpecification,
    output: &EstimationOutput,
) -> Result<StrategyBoundCausalEstimateCandidate, CausalError> {
    if output.kind == EstimationOutputKind::Associational {
        return Err(CausalError::AssociationalOutputNotCausal);
    }
    if strategy.estimand_id != estimand.id
        || estimator.estimand_id != estimand.id
        || output.estimand_id.as_deref() != Some(estimand.id.as_str())
    {
        return Err(CausalError::EstimandMismatch);
    }
    if estimator.strategy_id != strategy.id
        || output.strategy_id.as_deref() != Some(strategy.id.as_str())
    {
        return Err(CausalError::StrategyMismatch);
    }
    if output.estimator_id.as_deref() != Some(estimator.id.as_str()) {
        return Err(CausalError::EstimatorMismatch);
    }
    if output.treatment != estimand.treatment {
        return Err(CausalError::TreatmentMismatch);
    }
    if output.outcome != estimand.outcome {
        return Err(CausalError::OutcomeMismatch);
    }
    if output.outcome_unit != estimand.outcome_unit {
        return Err(CausalError::OutcomeUnitMismatch);
    }
    if output.lower_atoms > output.effect_atoms || output.effect_atoms > output.upper_atoms {
        return Err(CausalError::InvalidDesignContract);
    }

    Ok(StrategyBoundCausalEstimateCandidate {
        estimand_id: estimand.id.clone(),
        strategy_id: strategy.id.clone(),
        estimator_id: estimator.id.clone(),
        design_family: strategy.design.family_name(),
        assumptions: strategy.assumptions.clone(),
        diagnostic_artifact_ids: strategy.diagnostic_artifact_ids.clone(),
        estimator_family_id: estimator.estimator_family_id.clone(),
        implementation_artifact_id: estimator.implementation_artifact_id.clone(),
        uncertainty_method_artifact_id: estimator.uncertainty_method_artifact_id.clone(),
        effect_atoms: output.effect_atoms,
        lower_atoms: output.lower_atoms,
        upper_atoms: output.upper_atoms,
    })
}

fn policy_variable() -> EconomicVariable {
    EconomicVariable::new(
        VariableId::new("policy:treatment").unwrap(),
        StateDomain::Institutional,
        UnitId::new("binary").unwrap(),
        "Policy treatment assignment.",
    )
    .unwrap()
}

fn employment_variable() -> EconomicVariable {
    EconomicVariable::new(
        VariableId::new("labor:employment_rate").unwrap(),
        StateDomain::Institutional,
        UnitId::new("basis_points").unwrap(),
        "Employment rate in basis points.",
    )
    .unwrap()
}

fn estimand(id: &str, population: &str, horizon: &str) -> CausalEstimand {
    CausalEstimand::new(
        id,
        policy_variable().id().clone(),
        employment_variable().id().clone(),
        employment_variable().unit().clone(),
        population,
        "do(policy=1)-do(policy=0)",
        horizon,
        EffectTarget::AveragePopulation,
    )
    .unwrap()
}

fn randomized_strategy(estimand: &CausalEstimand) -> IdentificationStrategy {
    IdentificationStrategy::new(
        "identify:rct-v1",
        estimand,
        IdentificationDesign::RandomizedAssignment {
            assignment_artifact_id: "assignment:sealed-v1".into(),
            allocation_unit_id: "unit:household".into(),
        },
        vec![
            "assumption:no-interference-v1".into(),
            "assumption:randomization-integrity-v1".into(),
        ],
        vec!["diagnostic:balance-check-v1".into()],
    )
    .unwrap()
}

fn adjusted_strategy(estimand: &CausalEstimand) -> IdentificationStrategy {
    IdentificationStrategy::new(
        "identify:backdoor-v1",
        estimand,
        IdentificationDesign::BackdoorAdjustment {
            adjustment_set: vec![VariableId::new("baseline:income").unwrap()],
            causal_graph_artifact_id: "dag:policy-employment-v1".into(),
        },
        vec![
            "assumption:conditional-exchangeability-v1".into(),
            "assumption:positivity-v1".into(),
        ],
        vec!["diagnostic:overlap-v1".into()],
    )
    .unwrap()
}

fn estimator_for(
    estimand: &CausalEstimand,
    strategy: &IdentificationStrategy,
    id: &str,
) -> EstimatorSpecification {
    EstimatorSpecification::new(
        id,
        estimand,
        strategy,
        "estimator:difference-in-means-compatible-v1",
        "artifact:estimator-impl-v1",
        "artifact:bootstrap-ci-v1",
    )
    .unwrap()
}

fn strategy_bound_output(
    estimand: &CausalEstimand,
    strategy: &IdentificationStrategy,
    estimator: &EstimatorSpecification,
    effect_atoms: i128,
) -> EstimationOutput {
    EstimationOutput {
        kind: EstimationOutputKind::StrategyBound,
        estimand_id: Some(estimand.id.clone()),
        strategy_id: Some(strategy.id.clone()),
        estimator_id: Some(estimator.id.clone()),
        treatment: estimand.treatment.clone(),
        outcome: estimand.outcome.clone(),
        outcome_unit: estimand.outcome_unit.clone(),
        effect_atoms,
        lower_atoms: effect_atoms - 20,
        upper_atoms: effect_atoms + 20,
    }
}

#[test]
fn estimand_identity_includes_population_horizon_and_effect_target() {
    let national = estimand("estimand:national-v1", "population:national", "12_months");
    let youth = estimand("estimand:youth-v1", "population:youth", "12_months");
    let short = estimand("estimand:short-v1", "population:national", "3_months");
    let treated = CausalEstimand::new(
        "estimand:treated-v1",
        national.treatment.clone(),
        national.outcome.clone(),
        national.outcome_unit.clone(),
        national.population_id.clone(),
        national.intervention_contrast_id.clone(),
        national.horizon_id.clone(),
        EffectTarget::AverageTreated,
    )
    .unwrap();

    assert_ne!(national.population_id, youth.population_id);
    assert_ne!(national.horizon_id, short.horizon_id);
    assert_ne!(national.target, treated.target);
}

#[test]
fn strategy_requires_explicit_assumptions_and_design_contract() {
    let target = estimand("estimand:test-v1", "population:test", "1_year");
    assert_eq!(
        IdentificationStrategy::new(
            "identify:bad-v1",
            &target,
            IdentificationDesign::RandomizedAssignment {
                assignment_artifact_id: "".into(),
                allocation_unit_id: "unit:person".into(),
            },
            vec!["assumption:sutva-v1".into()],
            vec![],
        ),
        Err(CausalError::InvalidDesignContract)
    );
    assert_eq!(
        IdentificationStrategy::new(
            "identify:no-assumptions-v1",
            &target,
            IdentificationDesign::SyntheticControl {
                donor_pool_id: "donors:v1".into(),
                preperiod_fit_artifact_id: "fit:v1".into(),
            },
            vec![],
            vec![],
        ),
        Err(CausalError::EmptyAssumptions)
    );
}

#[test]
fn plain_association_cannot_upgrade_to_causal_estimate() {
    let target = estimand("estimand:rct-v1", "population:all", "1_year");
    let strategy = randomized_strategy(&target);
    let estimator = estimator_for(&target, &strategy, "estimator:rct-v1");
    let association = EstimationOutput {
        kind: EstimationOutputKind::Associational,
        estimand_id: None,
        strategy_id: None,
        estimator_id: None,
        treatment: target.treatment.clone(),
        outcome: target.outcome.clone(),
        outcome_unit: target.outcome_unit.clone(),
        effect_atoms: 75,
        lower_atoms: 50,
        upper_atoms: 100,
    };

    assert_eq!(
        qualify_causal_estimate(&target, &strategy, &estimator, &association),
        Err(CausalError::AssociationalOutputNotCausal)
    );
}

#[test]
fn identical_effect_values_from_different_strategies_remain_distinct_evidence() {
    let target = estimand("estimand:shared-v1", "population:all", "1_year");
    let randomized = randomized_strategy(&target);
    let adjusted = adjusted_strategy(&target);
    let randomized_estimator = estimator_for(&target, &randomized, "estimator:rct-v1");
    let adjusted_estimator = estimator_for(&target, &adjusted, "estimator:adjusted-v1");
    let randomized_output = strategy_bound_output(&target, &randomized, &randomized_estimator, 80);
    let adjusted_output = strategy_bound_output(&target, &adjusted, &adjusted_estimator, 80);

    let rct = qualify_causal_estimate(
        &target,
        &randomized,
        &randomized_estimator,
        &randomized_output,
    )
    .unwrap();
    let observational = qualify_causal_estimate(
        &target,
        &adjusted,
        &adjusted_estimator,
        &adjusted_output,
    )
    .unwrap();

    assert_eq!(rct.effect_atoms, observational.effect_atoms);
    assert_ne!(rct.design_family, observational.design_family);
    assert_ne!(rct.strategy_id, observational.strategy_id);
    assert_ne!(rct.assumptions, observational.assumptions);
}

#[test]
fn estimator_and_output_cannot_retarget_another_estimand() {
    let national = estimand("estimand:national-v1", "population:national", "1_year");
    let youth = estimand("estimand:youth-v1", "population:youth", "1_year");
    let strategy = randomized_strategy(&national);
    assert_eq!(
        EstimatorSpecification::new(
            "estimator:retarget-v1",
            &youth,
            &strategy,
            "estimator:mean-difference-v1",
            "artifact:impl-v1",
            "artifact:uncertainty-v1",
        ),
        Err(CausalError::EstimandMismatch)
    );

    let estimator = estimator_for(&national, &strategy, "estimator:national-v1");
    let mut output = strategy_bound_output(&national, &strategy, &estimator, 60);
    output.estimand_id = Some(youth.id.clone());
    assert_eq!(
        qualify_causal_estimate(&national, &strategy, &estimator, &output),
        Err(CausalError::EstimandMismatch)
    );
}

#[test]
fn treatment_outcome_and_unit_substitution_fail_closed() {
    let target = estimand("estimand:binding-v1", "population:all", "1_year");
    let strategy = randomized_strategy(&target);
    let estimator = estimator_for(&target, &strategy, "estimator:binding-v1");

    let mut wrong_treatment = strategy_bound_output(&target, &strategy, &estimator, 50);
    wrong_treatment.treatment = VariableId::new("policy:other-treatment").unwrap();
    assert_eq!(
        qualify_causal_estimate(&target, &strategy, &estimator, &wrong_treatment),
        Err(CausalError::TreatmentMismatch)
    );

    let mut wrong_outcome = strategy_bound_output(&target, &strategy, &estimator, 50);
    wrong_outcome.outcome = VariableId::new("labor:wages").unwrap();
    assert_eq!(
        qualify_causal_estimate(&target, &strategy, &estimator, &wrong_outcome),
        Err(CausalError::OutcomeMismatch)
    );

    let mut wrong_unit = strategy_bound_output(&target, &strategy, &estimator, 50);
    wrong_unit.outcome_unit = UnitId::new("percent").unwrap();
    assert_eq!(
        qualify_causal_estimate(&target, &strategy, &estimator, &wrong_unit),
        Err(CausalError::OutcomeUnitMismatch)
    );
}

#[test]
fn qualified_candidate_preserves_strategy_estimator_and_uncertainty_lineage() {
    let target = estimand("estimand:lineage-v1", "population:all", "1_year");
    let strategy = adjusted_strategy(&target);
    let estimator = estimator_for(&target, &strategy, "estimator:lineage-v1");
    let output = strategy_bound_output(&target, &strategy, &estimator, 40);
    let candidate = qualify_causal_estimate(&target, &strategy, &estimator, &output).unwrap();

    assert_eq!(candidate.estimand_id, target.id);
    assert_eq!(candidate.estimator_id, estimator.id);
    assert_eq!(candidate.design_family, "backdoor-adjustment");
    assert_eq!(candidate.diagnostic_artifact_ids, vec!["diagnostic:overlap-v1"]);
    assert_eq!(candidate.estimator_family_id, "estimator:difference-in-means-compatible-v1");
    assert_eq!(candidate.implementation_artifact_id, "artifact:estimator-impl-v1");
    assert_eq!(candidate.uncertainty_method_artifact_id, "artifact:bootstrap-ci-v1");
    assert_eq!((candidate.lower_atoms, candidate.effect_atoms, candidate.upper_atoms), (20, 40, 60));
}

#[test]
fn identification_families_are_not_encoded_as_a_truth_ranking() {
    let target = estimand("estimand:families-v1", "population:all", "1_year");
    let rct = randomized_strategy(&target);
    let adjusted = adjusted_strategy(&target);
    assert_ne!(rct.design.family_name(), adjusted.design.family_name());
    assert!(!rct.assumptions.is_empty());
    assert!(!adjusted.assumptions.is_empty());
}

#[test]
fn additional_design_families_require_their_own_explicit_contracts() {
    let target = estimand("estimand:designs-v1", "population:all", "1_year");
    let cases = [
        IdentificationDesign::InstrumentalVariable {
            instrument: VariableId::new("instrument:eligibility").unwrap(),
            first_stage_artifact_id: "first-stage:v1".into(),
            exclusion_assumption_id: "assumption:exclusion-v1".into(),
        },
        IdentificationDesign::RegressionDiscontinuity {
            running_variable: VariableId::new("running:score").unwrap(),
            cutoff_atoms: 500,
            continuity_assumption_id: "assumption:continuity-v1".into(),
        },
        IdentificationDesign::DifferenceInDifferences {
            group_definition_id: "groups:treated-control-v1".into(),
            intervention_time_id: "time:policy-start-v1".into(),
            parallel_trends_assumption_id: "assumption:parallel-trends-v1".into(),
        },
        IdentificationDesign::SyntheticControl {
            donor_pool_id: "donors:qualified-v1".into(),
            preperiod_fit_artifact_id: "fit:preperiod-v1".into(),
        },
    ];

    for (index, design) in cases.into_iter().enumerate() {
        let strategy = IdentificationStrategy::new(
            format!("identify:family-{index}"),
            &target,
            design,
            vec![format!("assumption:family-{index}")],
            vec![],
        )
        .unwrap();
        assert!(!strategy.design.family_name().is_empty());
    }
}

#[test]
fn conditional_and_local_estimands_require_named_target_population() {
    let treatment = policy_variable().id().clone();
    let outcome = employment_variable().id().clone();
    let unit = employment_variable().unit().clone();
    assert_eq!(
        CausalEstimand::new(
            "estimand:bad-conditional",
            treatment.clone(),
            outcome.clone(),
            unit.clone(),
            "population:all",
            "do(1)-do(0)",
            "1_year",
            EffectTarget::Conditional { subgroup_id: "".into() },
        ),
        Err(CausalError::EmptyText("conditional estimand subgroup"))
    );
    assert_eq!(
        CausalEstimand::new(
            "estimand:bad-local",
            treatment,
            outcome,
            unit,
            "population:all",
            "do(1)-do(0)",
            "1_year",
            EffectTarget::Local { population_id: " ".into() },
        ),
        Err(CausalError::EmptyText("local estimand population"))
    );
}
