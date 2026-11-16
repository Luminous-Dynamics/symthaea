//! Finite-ensemble, decision-focused value-of-information for fusion engineering.
//!
//! The producer owns scenario weights, action utilities, and experiment signal
//! mappings. This consumer does not infer probabilities, covariance, utility,
//! or test likelihoods. It computes exact finite-ensemble decision values over
//! the frozen joint input and preserves the supplied evidence class.

use std::collections::{BTreeMap, BTreeSet};

pub const DECISION_VOI_VERSION: &str = "fusion-decision-voi/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum DecisionEvidenceClass {
    SyntheticFixture,
    Assumption,
    ModelledOutput,
    MeasuredObservation,
    IndependentVerification,
    PhysicalQualification,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AlternativeEligibility {
    Eligible,
    BlockedByHardGate,
    UnknownHardGate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionScenarioWeight {
    pub scenario_id: String,
    pub probability_weight: f64,
    pub weight_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAlternative {
    pub alternative_id: String,
    pub eligibility: AlternativeEligibility,
    pub eligibility_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioUtility {
    pub scenario_id: String,
    pub alternative_id: String,
    /// Expected utility in the plan's single declared decision-value unit.
    pub utility_value: f64,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentSignal {
    pub scenario_id: String,
    pub signal_id: String,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InformationExperiment {
    pub experiment_id: String,
    pub experiment_design_sha256: String,
    /// Direct experiment cost in the same decision-value unit as utility.
    pub direct_cost_value: f64,
    /// Schedule, outage, and opportunity cost in that same unit.
    pub opportunity_cost_value: f64,
    pub cost_unit_ref: String,
    pub cost_basis_ref: String,
    pub cost_evidence_ref: String,
    pub evidence_ref: String,
    /// Optional variance-only acquisition baseline score. It is reported but
    /// never used in EVSI or NetVoI calculations.
    pub predictive_variance_score: Option<f64>,
    pub variance_score_evidence_ref: Option<String>,
    /// Complete deterministic mapping from every scenario to one possible
    /// observation signal. Probabilities are induced by scenario weights.
    pub signal_map: Vec<ExperimentSignal>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionValuePlan {
    pub decision_case_id: String,
    pub decision_case_sha256: String,
    pub scenario_weight_manifest_ref: String,
    pub scenario_weight_manifest_sha256: String,
    pub utility_manifest_ref: String,
    pub utility_manifest_sha256: String,
    pub cost_basis_ref: String,
    pub decision_value_unit_ref: String,
    pub lifetime_horizon_ref: String,
    pub validity_domain_ref: String,
    /// Refers to the source-owned finite joint scenario ensemble, not to an
    /// assumed independent set of marginal distributions.
    pub joint_scenario_model_ref: String,
    pub evidence_class: DecisionEvidenceClass,
    /// Allowed absolute deviation of the supplied scenario weights from one.
    pub weight_sum_tolerance: f64,
    /// Numerical tolerance for EVSI/EVPI/regret arithmetic.
    pub numerical_tolerance: f64,
    pub scenarios: Vec<DecisionScenarioWeight>,
    pub alternatives: Vec<DecisionAlternative>,
    pub utilities: Vec<ScenarioUtility>,
    pub experiments: Vec<InformationExperiment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionVoIDisposition {
    NotEvaluable,
    NoPositiveValueExperiment,
    SelectExperiment,
    SyntheticOnly,
    AssumptionOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum DecisionVoIDiagnostic {
    EmptyDecisionCaseIdentity,
    InvalidDecisionCaseDigest,
    EmptyScenarioWeightManifest,
    InvalidScenarioWeightDigest,
    EmptyUtilityManifest,
    InvalidUtilityManifestDigest,
    EmptyCostBasis,
    EmptyDecisionValueUnit,
    EmptyLifetimeHorizon,
    EmptyValidityDomain,
    EmptyJointScenarioModel,
    InvalidWeightSumTolerance,
    InvalidNumericalTolerance,
    NoScenarios,
    DuplicateScenarioIdentity,
    EmptyScenarioIdentity,
    InvalidScenarioWeight,
    EmptyScenarioWeightEvidence,
    ScenarioWeightsDoNotSumToOne,
    NoDecisionAlternatives,
    NoEligibleAlternative,
    DuplicateAlternativeIdentity,
    EmptyAlternativeIdentity,
    EmptyAlternativeEvidence,
    DuplicateUtilityEntry,
    UnknownUtilityScenario,
    UnknownUtilityAlternative,
    UtilityForIneligibleAlternative,
    MissingUtilityEntry,
    NonFiniteUtility,
    EmptyUtilityEvidence,
    NoExperiments,
    DuplicateExperimentIdentity,
    EmptyExperimentIdentity,
    InvalidExperimentDesignDigest,
    NonFiniteExperimentCost,
    NegativeExperimentCost,
    ExperimentCostUnitMismatch,
    ExperimentCostBasisMismatch,
    EmptyExperimentCostEvidence,
    EmptyExperimentEvidence,
    InvalidPredictiveVarianceScore,
    MissingPredictiveVarianceEvidence,
    DuplicateScenarioSignal,
    UnknownSignalScenario,
    MissingScenarioSignal,
    EmptySignalIdentity,
    EmptySignalEvidence,
    NonFiniteCurrentValue,
    InvalidEvpi,
    NegativeEvsi,
    EvsiExceedsEvpi,
    NegativeExpectedRegret,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SignalDecisionValue {
    pub signal_id: String,
    pub signal_probability: f64,
    pub selected_alternative_id: String,
    pub conditional_expected_utility: f64,
    pub weighted_post_signal_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentValueReport {
    pub experiment_id: String,
    pub direct_cost_value: f64,
    pub opportunity_cost_value: f64,
    pub total_cost_value: f64,
    pub predictive_variance_score: Option<f64>,
    pub expected_value_after_information: f64,
    pub evsi: f64,
    pub evpi: f64,
    pub expected_regret_after_information: f64,
    pub net_voi: f64,
    pub signals: Vec<SignalDecisionValue>,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionValueReport {
    pub contract_version: &'static str,
    pub decision_case_id: String,
    pub disposition: DecisionVoIDisposition,
    pub diagnostics: Vec<DecisionVoIDiagnostic>,
    pub evidence_class: DecisionEvidenceClass,
    pub decision_value_unit_ref: String,
    pub current_best_alternative_id: Option<String>,
    pub current_best_expected_value: Option<f64>,
    pub expected_value_with_perfect_information: Option<f64>,
    pub evpi: Option<f64>,
    pub experiments: Vec<ExperimentValueReport>,
    pub selected_experiment_id: Option<String>,
    pub selected_net_voi: Option<f64>,
    pub evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Evaluate EVSI/EVPI over an exact finite joint-scenario ensemble.
///
/// For each experiment, scenarios are grouped by the supplied deterministic
/// signal. Conditional expected utility is calculated for every eligible
/// alternative, and the best action is selected for each signal. No prior
/// weights are renormalized; no missing outcome is imputed.
pub fn evaluate_decision_value_of_information(
    plan: &DecisionValuePlan,
) -> DecisionValueReport {
    let mut diagnostics = validate_plan(plan);
    if !diagnostics.is_empty() {
        diagnostics.sort();
        diagnostics.dedup();
        return invalid_report(plan, diagnostics, collect_evidence_refs(plan));
    }

    let scenarios = scenario_map(plan);
    let alternatives = eligible_alternative_ids(plan);
    let utilities = utility_map(plan);
    let evidence_refs = collect_evidence_refs(plan);

    let mut current_values = BTreeMap::new();
    for alternative_id in &alternatives {
        let value = scenarios
            .iter()
            .map(|(scenario_id, scenario)| {
                scenario.probability_weight
                    * utilities[&(scenario_id.clone(), alternative_id.clone())]
            })
            .sum::<f64>();
        current_values.insert(alternative_id.clone(), value);
    }
    let (current_best_alternative_id, current_best_value) =
        select_best_alternative(&current_values);

    let mut perfect_information_value = 0.0;
    for (scenario_id, scenario) in &scenarios {
        let scenario_best = alternatives
            .iter()
            .map(|alternative_id| {
                utilities[&(scenario_id.clone(), alternative_id.clone())]
            })
            .max_by(f64::total_cmp)
            .unwrap_or(f64::NEG_INFINITY);
        perfect_information_value += scenario.probability_weight * scenario_best;
    }

    if !current_best_value.is_finite() || !perfect_information_value.is_finite() {
        diagnostics.push(DecisionVoIDiagnostic::NonFiniteCurrentValue);
        diagnostics.sort();
        diagnostics.dedup();
        return invalid_report(plan, diagnostics, evidence_refs);
    }

    let evpi = perfect_information_value - current_best_value;
    if evpi < -plan.numerical_tolerance {
        diagnostics.push(DecisionVoIDiagnostic::InvalidEvpi);
    }

    let mut experiment_reports = Vec::with_capacity(plan.experiments.len());
    for experiment in &plan.experiments {
        let signals = signal_map(experiment);
        let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for scenario_id in scenarios.keys() {
            if let Some(signal_id) = signals.get(scenario_id) {
                grouped
                    .entry(signal_id.clone())
                    .or_default()
                    .push(scenario_id.clone());
            }
        }

        let mut signal_reports = Vec::with_capacity(grouped.len());
        let mut expected_value_after_information = 0.0;
        for (signal_id, scenario_ids) in grouped {
            let signal_probability = scenario_ids
                .iter()
                .map(|scenario_id| scenarios[scenario_id].probability_weight)
                .sum::<f64>();
            // Zero-probability signals carry no posterior mass and are not
            // evaluated. Their scenarios still require a complete signal map.
            if signal_probability <= 0.0 {
                continue;
            }

            let mut conditional_values = BTreeMap::new();
            for alternative_id in &alternatives {
                let conditional_value = scenario_ids
                    .iter()
                    .map(|scenario_id| {
                        scenarios[scenario_id].probability_weight
                            * utilities[&(scenario_id.clone(), alternative_id.clone())]
                    })
                    .sum::<f64>()
                    / signal_probability;
                conditional_values.insert(alternative_id.clone(), conditional_value);
            }

            let (selected_alternative_id, conditional_expected_utility) =
                select_best_alternative(&conditional_values);
            let weighted_post_signal_value =
                signal_probability * conditional_expected_utility;
            expected_value_after_information += weighted_post_signal_value;
            signal_reports.push(SignalDecisionValue {
                signal_id,
                signal_probability,
                selected_alternative_id,
                conditional_expected_utility,
                weighted_post_signal_value,
            });
        }
        signal_reports.sort_by(|first, second| first.signal_id.cmp(&second.signal_id));

        let evsi = expected_value_after_information - current_best_value;
        let expected_regret_after_information =
            perfect_information_value - expected_value_after_information;
        let total_cost_value =
            experiment.direct_cost_value + experiment.opportunity_cost_value;
        let net_voi = evsi - total_cost_value;
        if !total_cost_value.is_finite()
            || !expected_value_after_information.is_finite()
            || !evsi.is_finite()
            || !expected_regret_after_information.is_finite()
            || !net_voi.is_finite()
        {
            diagnostics.push(DecisionVoIDiagnostic::NonFiniteCurrentValue);
        }

        if evsi < -plan.numerical_tolerance {
            diagnostics.push(DecisionVoIDiagnostic::NegativeEvsi);
        }
        if evsi > evpi + plan.numerical_tolerance {
            diagnostics.push(DecisionVoIDiagnostic::EvsiExceedsEvpi);
        }
        if expected_regret_after_information < -plan.numerical_tolerance {
            diagnostics.push(DecisionVoIDiagnostic::NegativeExpectedRegret);
        }

        let mut refs = vec![
            experiment.experiment_design_sha256.clone(),
            experiment.cost_evidence_ref.clone(),
            experiment.evidence_ref.clone(),
        ];
        refs.extend(
            experiment
                .variance_score_evidence_ref
                .iter()
                .cloned(),
        );
        refs.extend(
            experiment
                .signal_map
                .iter()
                .map(|signal| signal.evidence_ref.clone()),
        );
        refs.sort();
        refs.dedup();

        experiment_reports.push(ExperimentValueReport {
            experiment_id: experiment.experiment_id.clone(),
            direct_cost_value: experiment.direct_cost_value,
            opportunity_cost_value: experiment.opportunity_cost_value,
            total_cost_value,
            predictive_variance_score: experiment.predictive_variance_score,
            expected_value_after_information,
            evsi,
            evpi,
            expected_regret_after_information,
            net_voi,
            signals: signal_reports,
            evidence_refs: refs,
        });
    }

    if !diagnostics.is_empty() {
        diagnostics.sort();
        diagnostics.dedup();
        let mut report = invalid_report(plan, diagnostics, evidence_refs);
        report.current_best_alternative_id = Some(current_best_alternative_id);
        report.current_best_expected_value = Some(current_best_value);
        report.expected_value_with_perfect_information = Some(perfect_information_value);
        report.evpi = Some(evpi.max(0.0));
        report.experiments = experiment_reports;
        return report;
    }

    experiment_reports.sort_by(|first, second| {
        second
            .net_voi
            .total_cmp(&first.net_voi)
            .then_with(|| first.experiment_id.cmp(&second.experiment_id))
    });

    let best_experiment = experiment_reports
        .first()
        .filter(|experiment| experiment.net_voi > plan.numerical_tolerance);
    let selected_experiment_id = best_experiment.map(|experiment| experiment.experiment_id.clone());
    let selected_net_voi = best_experiment.map(|experiment| experiment.net_voi);

    let disposition = match plan.evidence_class {
        DecisionEvidenceClass::SyntheticFixture => DecisionVoIDisposition::SyntheticOnly,
        DecisionEvidenceClass::Assumption => DecisionVoIDisposition::AssumptionOnly,
        DecisionEvidenceClass::ModelledOutput
        | DecisionEvidenceClass::MeasuredObservation
        | DecisionEvidenceClass::IndependentVerification
        | DecisionEvidenceClass::PhysicalQualification => {
            if selected_experiment_id.is_some() {
                DecisionVoIDisposition::SelectExperiment
            } else {
                DecisionVoIDisposition::NoPositiveValueExperiment
            }
        }
    };

    DecisionValueReport {
        contract_version: DECISION_VOI_VERSION,
        decision_case_id: plan.decision_case_id.clone(),
        disposition,
        diagnostics,
        evidence_class: plan.evidence_class,
        decision_value_unit_ref: plan.decision_value_unit_ref.clone(),
        current_best_alternative_id: Some(current_best_alternative_id),
        current_best_expected_value: Some(current_best_value),
        expected_value_with_perfect_information: Some(perfect_information_value),
        evpi: Some(evpi.max(0.0)),
        experiments: experiment_reports,
        selected_experiment_id,
        selected_net_voi,
        evidence_refs,
        claim_ceiling: "finite-ensemble decision-value calculation over supplied weights, utilities and signals only",
    }
}

fn validate_plan(plan: &DecisionValuePlan) -> Vec<DecisionVoIDiagnostic> {
    let mut diagnostics = Vec::new();

    if plan.decision_case_id.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyDecisionCaseIdentity);
    }
    if !is_sha256(&plan.decision_case_sha256) {
        diagnostics.push(DecisionVoIDiagnostic::InvalidDecisionCaseDigest);
    }
    if plan.scenario_weight_manifest_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyScenarioWeightManifest);
    }
    if !is_sha256(&plan.scenario_weight_manifest_sha256) {
        diagnostics.push(DecisionVoIDiagnostic::InvalidScenarioWeightDigest);
    }
    if plan.utility_manifest_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyUtilityManifest);
    }
    if !is_sha256(&plan.utility_manifest_sha256) {
        diagnostics.push(DecisionVoIDiagnostic::InvalidUtilityManifestDigest);
    }
    if plan.cost_basis_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyCostBasis);
    }
    if plan.decision_value_unit_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyDecisionValueUnit);
    }
    if plan.lifetime_horizon_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyLifetimeHorizon);
    }
    if plan.validity_domain_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyValidityDomain);
    }
    if plan.joint_scenario_model_ref.trim().is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::EmptyJointScenarioModel);
    }
    if !plan.weight_sum_tolerance.is_finite() || plan.weight_sum_tolerance < 0.0 {
        diagnostics.push(DecisionVoIDiagnostic::InvalidWeightSumTolerance);
    }
    if !plan.numerical_tolerance.is_finite() || plan.numerical_tolerance < 0.0 {
        diagnostics.push(DecisionVoIDiagnostic::InvalidNumericalTolerance);
    }
    if plan.scenarios.is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::NoScenarios);
    }
    if plan.alternatives.is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::NoDecisionAlternatives);
    }
    if plan.experiments.is_empty() {
        diagnostics.push(DecisionVoIDiagnostic::NoExperiments);
    }

    let mut scenario_ids = BTreeSet::new();
    let mut weight_sum = 0.0;
    for scenario in &plan.scenarios {
        if scenario.scenario_id.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyScenarioIdentity);
        }
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            diagnostics.push(DecisionVoIDiagnostic::DuplicateScenarioIdentity);
        }
        if !scenario.probability_weight.is_finite() || scenario.probability_weight < 0.0 {
            diagnostics.push(DecisionVoIDiagnostic::InvalidScenarioWeight);
        }
        if scenario.weight_evidence_ref.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyScenarioWeightEvidence);
        }
        weight_sum += scenario.probability_weight;
    }
    if !plan.scenarios.is_empty()
        && weight_sum.is_finite()
        && (weight_sum - 1.0).abs() > plan.weight_sum_tolerance
    {
        diagnostics.push(DecisionVoIDiagnostic::ScenarioWeightsDoNotSumToOne);
    }

    let mut alternatives = BTreeMap::new();
    for alternative in &plan.alternatives {
        if alternative.alternative_id.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyAlternativeIdentity);
        }
        if alternative.eligibility_evidence_ref.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyAlternativeEvidence);
        }
        if alternatives
            .insert(alternative.alternative_id.as_str(), alternative.eligibility)
            .is_some()
        {
            diagnostics.push(DecisionVoIDiagnostic::DuplicateAlternativeIdentity);
        }
    }
    if !plan
        .alternatives
        .iter()
        .any(|alternative| alternative.eligibility == AlternativeEligibility::Eligible)
    {
        diagnostics.push(DecisionVoIDiagnostic::NoEligibleAlternative);
    }

    let mut utility_keys = BTreeSet::new();
    for utility in &plan.utilities {
        if !scenario_ids.contains(utility.scenario_id.as_str()) {
            diagnostics.push(DecisionVoIDiagnostic::UnknownUtilityScenario);
        }
        match alternatives.get(utility.alternative_id.as_str()) {
            None => diagnostics.push(DecisionVoIDiagnostic::UnknownUtilityAlternative),
            Some(eligibility) if *eligibility != AlternativeEligibility::Eligible => {
                diagnostics.push(DecisionVoIDiagnostic::UtilityForIneligibleAlternative);
            }
            Some(_) => {}
        }
        if !utility.utility_value.is_finite() {
            diagnostics.push(DecisionVoIDiagnostic::NonFiniteUtility);
        }
        if utility.evidence_ref.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyUtilityEvidence);
        }
        if !utility_keys.insert((utility.scenario_id.as_str(), utility.alternative_id.as_str())) {
            diagnostics.push(DecisionVoIDiagnostic::DuplicateUtilityEntry);
        }
    }

    for scenario_id in &scenario_ids {
        for (alternative_id, eligibility) in &alternatives {
            if *eligibility == AlternativeEligibility::Eligible
                && !utility_keys.contains(&(*scenario_id, *alternative_id))
            {
                diagnostics.push(DecisionVoIDiagnostic::MissingUtilityEntry);
            }
        }
    }

    let mut experiment_ids = BTreeSet::new();
    for experiment in &plan.experiments {
        if experiment.experiment_id.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyExperimentIdentity);
        }
        if !experiment_ids.insert(experiment.experiment_id.as_str()) {
            diagnostics.push(DecisionVoIDiagnostic::DuplicateExperimentIdentity);
        }
        if !is_sha256(&experiment.experiment_design_sha256) {
            diagnostics.push(DecisionVoIDiagnostic::InvalidExperimentDesignDigest);
        }
        if !experiment.direct_cost_value.is_finite()
            || !experiment.opportunity_cost_value.is_finite()
        {
            diagnostics.push(DecisionVoIDiagnostic::NonFiniteExperimentCost);
        }
        if experiment.direct_cost_value < 0.0 || experiment.opportunity_cost_value < 0.0 {
            diagnostics.push(DecisionVoIDiagnostic::NegativeExperimentCost);
        }
        if experiment.cost_unit_ref != plan.decision_value_unit_ref {
            diagnostics.push(DecisionVoIDiagnostic::ExperimentCostUnitMismatch);
        }
        if experiment.cost_basis_ref != plan.cost_basis_ref {
            diagnostics.push(DecisionVoIDiagnostic::ExperimentCostBasisMismatch);
        }
        if experiment.cost_evidence_ref.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyExperimentCostEvidence);
        }
        if experiment.evidence_ref.trim().is_empty() {
            diagnostics.push(DecisionVoIDiagnostic::EmptyExperimentEvidence);
        }
        if experiment
            .predictive_variance_score
            .is_some_and(|score| !score.is_finite() || score < 0.0)
        {
            diagnostics.push(DecisionVoIDiagnostic::InvalidPredictiveVarianceScore);
        }
        if experiment.predictive_variance_score.is_some()
            && experiment
                .variance_score_evidence_ref
                .as_deref()
                .is_none_or(str::is_empty)
        {
            diagnostics.push(DecisionVoIDiagnostic::MissingPredictiveVarianceEvidence);
        }
        if experiment.predictive_variance_score.is_none()
            && experiment.variance_score_evidence_ref.is_some()
        {
            diagnostics.push(DecisionVoIDiagnostic::MissingPredictiveVarianceEvidence);
        }

        let mut mapped_scenarios = BTreeSet::new();
        for signal in &experiment.signal_map {
            if !scenario_ids.contains(signal.scenario_id.as_str()) {
                diagnostics.push(DecisionVoIDiagnostic::UnknownSignalScenario);
            }
            if !mapped_scenarios.insert(signal.scenario_id.as_str()) {
                diagnostics.push(DecisionVoIDiagnostic::DuplicateScenarioSignal);
            }
            if signal.signal_id.trim().is_empty() {
                diagnostics.push(DecisionVoIDiagnostic::EmptySignalIdentity);
            }
            if signal.evidence_ref.trim().is_empty() {
                diagnostics.push(DecisionVoIDiagnostic::EmptySignalEvidence);
            }
        }
        if mapped_scenarios.len() != scenario_ids.len()
            || !scenario_ids.iter().all(|scenario_id| mapped_scenarios.contains(scenario_id))
        {
            diagnostics.push(DecisionVoIDiagnostic::MissingScenarioSignal);
        }
    }

    diagnostics.sort();
    diagnostics.dedup();
    diagnostics
}

fn scenario_map(plan: &DecisionValuePlan) -> BTreeMap<String, &DecisionScenarioWeight> {
    plan.scenarios
        .iter()
        .map(|scenario| (scenario.scenario_id.clone(), scenario))
        .collect()
}

fn eligible_alternative_ids(plan: &DecisionValuePlan) -> Vec<String> {
    let mut ids: Vec<String> = plan
        .alternatives
        .iter()
        .filter(|alternative| alternative.eligibility == AlternativeEligibility::Eligible)
        .map(|alternative| alternative.alternative_id.clone())
        .collect();
    ids.sort();
    ids
}

fn utility_map(plan: &DecisionValuePlan) -> BTreeMap<(String, String), f64> {
    plan.utilities
        .iter()
        .map(|utility| {
            (
                (utility.scenario_id.clone(), utility.alternative_id.clone()),
                utility.utility_value,
            )
        })
        .collect()
}

fn signal_map(experiment: &InformationExperiment) -> BTreeMap<String, String> {
    experiment
        .signal_map
        .iter()
        .map(|signal| (signal.scenario_id.clone(), signal.signal_id.clone()))
        .collect()
}

fn select_best_alternative(values: &BTreeMap<String, f64>) -> (String, f64) {
    let mut iter = values.iter();
    let (first_id, first_value) = iter.next().expect("validated eligible alternative set");
    let mut selected_id = first_id.clone();
    let mut selected_value = *first_value;
    for (alternative_id, value) in iter {
        if *value > selected_value {
            selected_id = alternative_id.clone();
            selected_value = *value;
        }
    }
    (selected_id, selected_value)
}

fn collect_evidence_refs(plan: &DecisionValuePlan) -> Vec<String> {
    let mut refs = vec![
        plan.decision_case_sha256.clone(),
        plan.scenario_weight_manifest_ref.clone(),
        plan.scenario_weight_manifest_sha256.clone(),
        plan.utility_manifest_ref.clone(),
        plan.utility_manifest_sha256.clone(),
        plan.joint_scenario_model_ref.clone(),
    ];
    refs.extend(
        plan.scenarios
            .iter()
            .map(|scenario| scenario.weight_evidence_ref.clone()),
    );
    refs.extend(
        plan.alternatives
            .iter()
            .map(|alternative| alternative.eligibility_evidence_ref.clone()),
    );
    refs.extend(plan.utilities.iter().map(|utility| utility.evidence_ref.clone()));
    refs.extend(plan.experiments.iter().flat_map(|experiment| {
        std::iter::once(experiment.experiment_design_sha256.clone())
            .chain(std::iter::once(experiment.cost_evidence_ref.clone()))
            .chain(std::iter::once(experiment.evidence_ref.clone()))
            .chain(experiment.variance_score_evidence_ref.iter().cloned())
            .chain(
                experiment
                    .signal_map
                    .iter()
                    .map(|signal| signal.evidence_ref.clone()),
            )
    }));
    refs.sort();
    refs.dedup();
    refs
}

fn invalid_report(
    plan: &DecisionValuePlan,
    diagnostics: Vec<DecisionVoIDiagnostic>,
    evidence_refs: Vec<String>,
) -> DecisionValueReport {
    DecisionValueReport {
        contract_version: DECISION_VOI_VERSION,
        decision_case_id: plan.decision_case_id.clone(),
        disposition: DecisionVoIDisposition::NotEvaluable,
        diagnostics,
        evidence_class: plan.evidence_class,
        decision_value_unit_ref: plan.decision_value_unit_ref.clone(),
        current_best_alternative_id: None,
        current_best_expected_value: None,
        expected_value_with_perfect_information: None,
        evpi: None,
        experiments: Vec::new(),
        selected_experiment_id: None,
        selected_net_voi: None,
        evidence_refs,
        claim_ceiling: "finite-ensemble decision-value calculation over supplied weights, utilities and signals only",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn scenario(id: &str, weight: f64) -> DecisionScenarioWeight {
        DecisionScenarioWeight {
            scenario_id: id.to_string(),
            probability_weight: weight,
            weight_evidence_ref: format!("weight-evidence-{id}"),
        }
    }

    fn alternative(id: &str) -> DecisionAlternative {
        DecisionAlternative {
            alternative_id: id.to_string(),
            eligibility: AlternativeEligibility::Eligible,
            eligibility_evidence_ref: format!("eligibility-evidence-{id}"),
        }
    }

    fn utility(scenario_id: &str, alternative_id: &str, value: f64) -> ScenarioUtility {
        ScenarioUtility {
            scenario_id: scenario_id.to_string(),
            alternative_id: alternative_id.to_string(),
            utility_value: value,
            evidence_ref: format!("utility-{scenario_id}-{alternative_id}"),
        }
    }

    fn experiment(
        id: &str,
        direct_cost: f64,
        opportunity_cost: f64,
        signal_groups: &[(&str, &str)],
    ) -> InformationExperiment {
        InformationExperiment {
            experiment_id: id.to_string(),
            experiment_design_sha256: sha('d'),
            direct_cost_value: direct_cost,
            opportunity_cost_value: opportunity_cost,
            cost_unit_ref: "USD-2026-real-million".to_string(),
            cost_basis_ref: "USD-2026-real-discount-6pct-v1".to_string(),
            cost_evidence_ref: format!("cost-evidence-{id}"),
            evidence_ref: format!("experiment-evidence-{id}"),
            predictive_variance_score: Some(predictive_variance_score),
            variance_score_evidence_ref: Some(format!("variance-score-{id}")),
            signal_map: signal_groups
                .iter()
                .map(|(scenario_id, signal_id)| ExperimentSignal {
                    scenario_id: (*scenario_id).to_string(),
                    signal_id: (*signal_id).to_string(),
                    evidence_ref: format!("signal-{id}-{scenario_id}-{signal_id}"),
                })
                .collect(),
        }
    }

    fn plan() -> DecisionValuePlan {
        DecisionValuePlan {
            decision_case_id: "finite-fusion-service-decision-v1".to_string(),
            decision_case_sha256: sha('a'),
            scenario_weight_manifest_ref: "scenario-weight-manifest-v1".to_string(),
            scenario_weight_manifest_sha256: sha('b'),
            utility_manifest_ref: "source-utility-manifest-v1".to_string(),
            utility_manifest_sha256: sha('c'),
            cost_basis_ref: "USD-2026-real-discount-6pct-v1".to_string(),
            decision_value_unit_ref: "USD-2026-real-million".to_string(),
            lifetime_horizon_ref: "plant-life-30-years-v1".to_string(),
            validity_domain_ref: "declared-demonstrator-service-domain-v1".to_string(),
            joint_scenario_model_ref: "joint-scenario-ensemble-v1".to_string(),
            evidence_class: DecisionEvidenceClass::SyntheticFixture,
            weight_sum_tolerance: 1e-12,
            numerical_tolerance: 1e-10,
            scenarios: vec![
                scenario("s1", 0.25),
                scenario("s2", 0.25),
                scenario("s3", 0.25),
                scenario("s4", 0.25),
            ],
            alternatives: vec![alternative("A"), alternative("B")],
            utilities: vec![
                utility("s1", "A", 100.0), utility("s1", "B", 60.0),
                utility("s2", "A", 90.0), utility("s2", "B", 75.0),
                utility("s3", "A", 40.0), utility("s3", "B", 95.0),
                utility("s4", "A", 20.0), utility("s4", "B", 110.0),
            ],
            experiments: vec![
                experiment(
                    "e1-decision-discriminating-test",
                    3.0,
                    2.0,
                    1.0,
                    &[("s1", "left"), ("s2", "left"), ("s3", "right"), ("s4", "right")],
                ),
                experiment(
                    "e2-high-variance-uninformative-test",
                    0.5,
                    0.5,
                    100.0,
                    &[("s1", "same"), ("s2", "same"), ("s3", "same"), ("s4", "same")],
                ),
                experiment(
                    "e3-overpriced-perfect-test",
                    20.0,
                    0.0,
                    25.0,
                    &[("s1", "s1"), ("s2", "s2"), ("s3", "s3"), ("s4", "s4")],
                ),
            ],
        }
    }

    #[test]
    fn known_answer_voi_selects_decision_discriminating_experiment() {
        let report = evaluate_decision_value_of_information(&plan());
        assert_eq!(report.disposition, DecisionVoIDisposition::SyntheticOnly);
        assert_eq!(report.current_best_alternative_id.as_deref(), Some("B"));
        assert!((report.current_best_expected_value.unwrap() - 85.0).abs() < 1e-12);
        assert!((report.expected_value_with_perfect_information.unwrap() - 98.75).abs() < 1e-12);
        assert!((report.evpi.unwrap() - 13.75).abs() < 1e-12);
        assert_eq!(
            report.experiments.first().unwrap().experiment_id,
            "e1-decision-discriminating-test"
        );
        assert!((report.experiments[0].evsi - 13.75).abs() < 1e-12);
        assert!((report.experiments[0].expected_regret_after_information).abs() < 1e-12);
        assert!((report.experiments[0].total_cost_value - 5.0).abs() < 1e-12);
        assert!((report.experiments[0].net_voi - 8.75).abs() < 1e-12);
        assert_eq!(
            report.selected_experiment_id.as_deref(),
            Some("e1-decision-discriminating-test")
        );
        assert_eq!(report.selected_net_voi, Some(8.75));
    }

    #[test]
    fn uninformative_high_variance_test_has_zero_evsi() {
        let report = evaluate_decision_value_of_information(&plan());
        let item = report
            .experiments
            .iter()
            .find(|item| item.experiment_id == "e2-high-variance-uninformative-test")
            .unwrap();
        assert_eq!(item.predictive_variance_score, Some(100.0));
        assert!(
            report.experiments[0].predictive_variance_score.unwrap()
                < item.predictive_variance_score.unwrap()
        );
        assert!(item.evsi.abs() < 1e-12);
        assert!((item.expected_regret_after_information - 13.75).abs() < 1e-12);
        assert!(item.net_voi < 0.0);
    }

    #[test]
    fn perfect_information_evsi_equals_evpi() {
        let report = evaluate_decision_value_of_information(&plan());
        let item = report
            .experiments
            .iter()
            .find(|item| item.experiment_id == "e3-overpriced-perfect-test")
            .unwrap();
        assert!((item.evsi - item.evpi).abs() < 1e-12);
        assert!(item.expected_regret_after_information.abs() < 1e-12);
        assert!(item.net_voi < 0.0);
    }

    #[test]
    fn experiment_order_does_not_change_decision_or_ranking() {
        let first = evaluate_decision_value_of_information(&plan());
        let mut reordered = plan();
        reordered.experiments.reverse();
        reordered.utilities.reverse();
        reordered.scenarios.reverse();
        let second = evaluate_decision_value_of_information(&reordered);
        assert_eq!(first, second);
    }

    #[test]
    fn scenario_weights_must_sum_to_one_without_renormalization() {
        let mut input = plan();
        input.scenarios[0].probability_weight = 0.2;
        let report = evaluate_decision_value_of_information(&input);
        assert_eq!(report.disposition, DecisionVoIDisposition::NotEvaluable);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::ScenarioWeightsDoNotSumToOne));
    }

    #[test]
    fn missing_action_utility_is_not_imputed_as_zero() {
        let mut input = plan();
        input
            .utilities
            .retain(|utility| !(utility.scenario_id == "s4" && utility.alternative_id == "A"));
        let report = evaluate_decision_value_of_information(&input);
        assert_eq!(report.disposition, DecisionVoIDisposition::NotEvaluable);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::MissingUtilityEntry));
    }

    #[test]
    fn duplicate_scenario_identity_is_refused() {
        let mut input = plan();
        input.scenarios.push(input.scenarios[0].clone());
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::DuplicateScenarioIdentity));
    }

    #[test]
    fn missing_experiment_signal_for_one_scenario_is_refused() {
        let mut input = plan();
        input.experiments[0]
            .signal_map
            .retain(|signal| signal.scenario_id != "s4");
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::MissingScenarioSignal));
    }

    #[test]
    fn duplicate_scenario_signal_is_refused() {
        let mut input = plan();
        input.experiments[0]
            .signal_map
            .push(input.experiments[0].signal_map[0].clone());
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::DuplicateScenarioSignal));
    }

    #[test]
    fn negative_experiment_cost_is_refused() {
        let mut input = plan();
        input.experiments[0].direct_cost_value = -1.0;
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::NegativeExperimentCost));
    }

    #[test]
    fn experiment_cost_unit_and_basis_must_match_utility() {
        let mut input = plan();
        input.experiments[0].cost_unit_ref = "MWh".to_string();
        input.experiments[1].cost_basis_ref = "USD-2025-nominal".to_string();
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::ExperimentCostUnitMismatch));
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::ExperimentCostBasisMismatch));
    }

    #[test]
    fn blocked_alternative_never_wins_decision() {
        let mut input = plan();
        input.alternatives.push(DecisionAlternative {
            alternative_id: "C-blocked".to_string(),
            eligibility: AlternativeEligibility::BlockedByHardGate,
            eligibility_evidence_ref: "blocked-gate-evidence".to_string(),
        });
        let report = evaluate_decision_value_of_information(&input);
        assert_eq!(report.current_best_alternative_id.as_deref(), Some("B"));
        assert_eq!(report.disposition, DecisionVoIDisposition::SyntheticOnly);
    }

    #[test]
    fn no_eligible_alternative_is_not_evaluable() {
        let mut input = plan();
        for alternative in &mut input.alternatives {
            alternative.eligibility = AlternativeEligibility::BlockedByHardGate;
        }
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::NoEligibleAlternative));
    }

    #[test]
    fn synthetic_ensemble_never_becomes_actionable_recommendation() {
        let report = evaluate_decision_value_of_information(&plan());
        assert_eq!(report.disposition, DecisionVoIDisposition::SyntheticOnly);
        assert!(report.selected_experiment_id.is_some());
    }

    #[test]
    fn assumption_only_ensemble_is_not_promoted() {
        let mut input = plan();
        input.evidence_class = DecisionEvidenceClass::Assumption;
        let report = evaluate_decision_value_of_information(&input);
        assert_eq!(report.disposition, DecisionVoIDisposition::AssumptionOnly);
    }

    #[test]
    fn no_positive_net_voi_returns_no_positive_experiment() {
        let mut input = plan();
        for experiment in &mut input.experiments {
            experiment.direct_cost_value += 1_000.0;
        }
        input.evidence_class = DecisionEvidenceClass::ModelledOutput;
        let report = evaluate_decision_value_of_information(&input);
        assert_eq!(report.disposition, DecisionVoIDisposition::NoPositiveValueExperiment);
        assert_eq!(report.selected_experiment_id, None);
    }

    #[test]
    fn duplicate_utility_and_experiment_ids_are_refused() {
        let mut input = plan();
        input.utilities.push(input.utilities[0].clone());
        input.experiments.push(input.experiments[0].clone());
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::DuplicateUtilityEntry));
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::DuplicateExperimentIdentity));
    }

    #[test]
    fn invalid_manifest_digest_or_signal_evidence_is_refused() {
        let mut input = plan();
        input.utility_manifest_sha256 = "bad".to_string();
        input.experiments[0].signal_map[0].evidence_ref.clear();
        let report = evaluate_decision_value_of_information(&input);
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::InvalidUtilityManifestDigest));
        assert!(report
            .diagnostics
            .contains(&DecisionVoIDiagnostic::EmptySignalEvidence));
    }

    #[test]
    fn zero_weight_scenario_is_allowed_but_zero_probability_signal_is_omitted() {
        let mut input = plan();
        input.scenarios[3].probability_weight = 0.0;
        input.scenarios[0].probability_weight = 1.0 / 3.0;
        input.scenarios[1].probability_weight = 1.0 / 3.0;
        input.scenarios[2].probability_weight = 1.0 / 3.0;
        // The full signal mapping is retained for s4; its signal carries zero
        // probability and is skipped in conditional posterior calculations.
        let report = evaluate_decision_value_of_information(&input);
        assert_eq!(report.disposition, DecisionVoIDisposition::SyntheticOnly);
        let perfect = report
            .experiments
            .iter()
            .find(|item| item.experiment_id == "e3-overpriced-perfect-test")
            .unwrap();
        assert!(perfect.signals.iter().all(|signal| signal.signal_probability > 0.0));
    }

    #[test]
    fn scenario_weight_order_does_not_change_evsi() {
        let input = plan();
        let first = evaluate_decision_value_of_information(&input);
        let mut reordered = input;
        reordered.scenarios.reverse();
        let second = evaluate_decision_value_of_information(&reordered);
        assert_eq!(first, second);
    }
}
