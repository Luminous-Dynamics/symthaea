//! Decision-focused experiment planning with explicit ScenarioImpact and calibrated EVSI modes.
//!
//! This module does not fit priors, infer probabilities, calibrate physical
//! models or recommend unexecuted experiments as evidence. When calibrated
//! probability/utility inputs are absent, it reports unweighted scenario
//! impact only and refuses to manufacture an EVSI number.

pub const EXPERIMENT_VALUE_VERSION: &str = "fusion-decision-experiment/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentMode {
    ScenarioImpact,
    CalibratedEvsi,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentFeasibility {
    Available,
    NotAvailable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionEligibility {
    Eligible,
    HardGateBlocked,
    HardGateUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentEvidenceClass {
    SyntheticFixture,
    Assumption,
    ModelledUncalibrated,
    CalibratedProbabilisticModel,
    ExecutedExperiment,
    IndependentlyVerifiedExperiment,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentAction {
    pub action_id: String,
    pub eligibility: ActionEligibility,
    /// Expected utility before the experiment, in the plan's frozen utility basis.
    /// Required for every eligible action in CalibratedEvsi mode.
    pub current_expected_utility: Option<f64>,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioImpactOutcome {
    pub outcome_id: String,
    /// Decision after this outcome. None means it remains unresolved.
    pub post_outcome_action_id: Option<String>,
    pub resolves_current_disagreement: bool,
    pub reveals_new_hard_gate_failure: bool,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalActionUtility {
    pub action_id: String,
    /// E[U(action, theta) | observed experiment outcome] in the frozen utility basis.
    pub expected_utility: f64,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbabilisticExperimentOutcome {
    pub outcome_id: String,
    pub probability: f64,
    /// Complete conditional expected utility for each eligible action.
    pub conditional_utilities: Vec<ConditionalActionUtility>,
    pub probability_evidence_ref: String,
    pub utility_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentValuePlan {
    pub experiment_id: String,
    pub experiment_revision: String,
    pub experiment_protocol_ref: String,
    pub experiment_protocol_sha256: String,
    pub decision_profile_ref: String,
    pub decision_profile_sha256: String,
    pub decision_envelope_ref: String,
    pub decision_envelope_sha256: String,
    pub utility_basis_ref: String,
    pub utility_basis_sha256: String,
    pub evidence_class: ExperimentEvidenceClass,
    pub feasibility: ExperimentFeasibility,
    pub duration_hours: f64,
    /// Monetary/facility cost is not mixed into utility implicitly. In EVSI
    /// mode this is the cost already converted to utility units under the
    /// exact utility_basis_ref and backed by cost_utility_evidence_ref.
    pub experiment_cost_utility_units: Option<f64>,
    pub cost_utility_basis_ref: Option<String>,
    pub cost_utility_evidence_ref: Option<String>,
    pub current_decision_action_id: Option<String>,
    pub current_decision_is_split: bool,
    pub actions: Vec<ExperimentAction>,
    pub mode: ExperimentMode,
    /// Required only for calibrated EVSI.
    pub probability_model_ref: Option<String>,
    pub probability_model_sha256: Option<String>,
    pub calibration_manifest_ref: Option<String>,
    pub calibration_manifest_sha256: Option<String>,
    pub joint_outcome_model_ref: Option<String>,
    pub joint_outcome_model_sha256: Option<String>,
    pub probability_sum_tolerance: f64,
    /// Unweighted alternatives: no probabilities are inferred.
    pub scenario_outcomes: Vec<ScenarioImpactOutcome>,
    /// Probability-weighted alternatives: only admitted in calibrated EVSI mode.
    pub probabilistic_outcomes: Vec<ProbabilisticExperimentOutcome>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentValueDisposition {
    ScenarioImpactOnly,
    CalibratedEvsiComputed,
    NotEvaluable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentValueDiagnostic {
    EmptyExperimentIdentity,
    EmptyProtocolIdentity,
    InvalidProtocolDigest,
    EmptyDecisionProfileIdentity,
    InvalidDecisionProfileDigest,
    EmptyDecisionEnvelopeIdentity,
    InvalidDecisionEnvelopeDigest,
    EmptyUtilityBasis,
    InvalidUtilityBasisDigest,
    InvalidDuration,
    MissingCurrentAction,
    UnknownCurrentAction,
    CurrentActionBlocked,
    DuplicateAction,
    EmptyActionIdentity,
    EmptyActionEvidence,
    UnknownActionEligibility,
    MissingCurrentUtility,
    NonFiniteCurrentUtility,
    MissingScenarioOutcomes,
    MissingScenarioOutcomeIdentity,
    DuplicateScenarioOutcome,
    EmptyScenarioOutcomeEvidence,
    UnknownPostOutcomeAction,
    PostOutcomeActionBlocked,
    InconsistentModeInputs,
    MissingProbabilityModel,
    InvalidProbabilityModelDigest,
    MissingCalibrationManifest,
    InvalidCalibrationManifestDigest,
    MissingJointOutcomeModel,
    InvalidJointOutcomeModelDigest,
    InvalidProbabilitySumTolerance,
    MissingProbabilisticOutcomes,
    MissingCostUtility,
    InvalidCostUtility,
    CostUtilityBasisMismatch,
    MissingCostEvidence,
    InvalidOutcomeProbability,
    ProbabilitySumMismatch,
    DuplicateProbabilisticOutcome,
    EmptyProbabilisticOutcomeIdentity,
    MissingProbabilityEvidence,
    MissingUtilityEvidence,
    MissingConditionalUtility,
    DuplicateConditionalUtility,
    UnknownConditionalUtilityAction,
    MissingConditionalUtilityEvidence,
    NonFiniteConditionalUtility,
    SyntheticEvsiRefused,
    AssumptionOnlyEvsiRefused,
    UncalibratedEvsiRefused,
    ExperimentInfeasible,
    ExperimentFeasibilityUnknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioImpactSummary {
    pub outcome_count: usize,
    pub unchanged_action_count: usize,
    pub changed_action_count: usize,
    pub unresolved_action_count: usize,
    pub resolves_current_disagreement_count: usize,
    pub reveals_hard_gate_failure_count: usize,
    /// Exact outcome-to-action mapping; no probabilities or weights.
    pub outcomes: Vec<ScenarioImpactOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvsiOutcomeSummary {
    pub outcome_id: String,
    pub probability: f64,
    pub best_action_id: String,
    pub best_conditional_expected_utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedEvsiSummary {
    pub current_best_action_id: String,
    pub current_best_expected_utility: f64,
    pub expected_post_experiment_best_utility: f64,
    pub gross_evsi_utility_units: f64,
    pub experiment_cost_utility_units: f64,
    pub net_evsi_utility_units: f64,
    pub decision_change_probability: f64,
    pub outcomes: Vec<EvsiOutcomeSummary>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentValueReport {
    pub contract_version: &'static str,
    pub experiment_id: String,
    pub experiment_revision: String,
    pub mode: ExperimentMode,
    pub disposition: ExperimentValueDisposition,
    pub feasibility: ExperimentFeasibility,
    pub duration_hours: f64,
    pub scenario_impact: Option<ScenarioImpactSummary>,
    pub calibrated_evsi: Option<CalibratedEvsiSummary>,
    pub diagnostics: Vec<ExperimentValueDiagnostic>,
    pub evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

/// Evaluate an experiment plan. Calibrated EVSI is never inferred from
/// scenario counts, model spread, parameter variance or uncalibrated
/// probabilities; it requires an explicit joint outcome/calibration contract.
pub fn evaluate_experiment_value(plan: &ExperimentValuePlan) -> ExperimentValueReport {
    let mut diagnostics = validate_common(plan);
    let evidence_refs = collect_evidence_refs(plan);

    if !diagnostics.is_empty() {
        return invalid_report(plan, diagnostics, evidence_refs);
    }

    match plan.mode {
        ExperimentMode::ScenarioImpact => {
            if !plan.probabilistic_outcomes.is_empty()
                || plan.probability_model_ref.is_some()
                || plan.calibration_manifest_ref.is_some()
                || plan.joint_outcome_model_ref.is_some()
                || plan.experiment_cost_utility_units.is_some()
            {
                diagnostics.push(ExperimentValueDiagnostic::InconsistentModeInputs);
                return invalid_report(plan, diagnostics, evidence_refs);
            }
            let Some(summary) = evaluate_scenario_impact(plan, &mut diagnostics) else {
                return invalid_report(plan, diagnostics, evidence_refs);
            };
            if !diagnostics.is_empty() {
                return invalid_report(plan, diagnostics, evidence_refs);
            }
            ExperimentValueReport {
                contract_version: EXPERIMENT_VALUE_VERSION,
                experiment_id: plan.experiment_id.clone(),
                experiment_revision: plan.experiment_revision.clone(),
                mode: plan.mode,
                disposition: ExperimentValueDisposition::ScenarioImpactOnly,
                feasibility: plan.feasibility,
                duration_hours: plan.duration_hours,
                scenario_impact: Some(summary),
                calibrated_evsi: None,
                diagnostics,
                evidence_refs,
                claim_ceiling: "unweighted finite experiment-outcome decision-impact report only",
            }
        }
        ExperimentMode::CalibratedEvsi => {
            let Some(summary) = evaluate_calibrated_evsi(plan, &mut diagnostics) else {
                return invalid_report(plan, diagnostics, evidence_refs);
            };
            if !diagnostics.is_empty() {
                return invalid_report(plan, diagnostics, evidence_refs);
            }
            ExperimentValueReport {
                contract_version: EXPERIMENT_VALUE_VERSION,
                experiment_id: plan.experiment_id.clone(),
                experiment_revision: plan.experiment_revision.clone(),
                mode: plan.mode,
                disposition: ExperimentValueDisposition::CalibratedEvsiComputed,
                feasibility: plan.feasibility,
                duration_hours: plan.duration_hours,
                scenario_impact: None,
                calibrated_evsi: Some(summary),
                diagnostics,
                evidence_refs,
                claim_ceiling: "EVSI under the exact supplied calibrated probability and utility model only",
            }
        }
    }
}

fn validate_common(plan: &ExperimentValuePlan) -> Vec<ExperimentValueDiagnostic> {
    let mut diagnostics = Vec::new();
    if plan.experiment_id.trim().is_empty() || plan.experiment_revision.trim().is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::EmptyExperimentIdentity);
    }
    if plan.experiment_protocol_ref.trim().is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::EmptyProtocolIdentity);
    }
    if !is_sha256(&plan.experiment_protocol_sha256) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidProtocolDigest);
    }
    if plan.decision_profile_ref.trim().is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::EmptyDecisionProfileIdentity);
    }
    if !is_sha256(&plan.decision_profile_sha256) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidDecisionProfileDigest);
    }
    if plan.decision_envelope_ref.trim().is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::EmptyDecisionEnvelopeIdentity);
    }
    if !is_sha256(&plan.decision_envelope_sha256) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidDecisionEnvelopeDigest);
    }
    if plan.utility_basis_ref.trim().is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::EmptyUtilityBasis);
    }
    if !is_sha256(&plan.utility_basis_sha256) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidUtilityBasisDigest);
    }
    if !plan.duration_hours.is_finite() || plan.duration_hours <= 0.0 {
        diagnostics.push(ExperimentValueDiagnostic::InvalidDuration);
    }
    if plan.actions.is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::MissingCurrentAction);
    }

    let mut action_ids = std::collections::BTreeSet::new();
    for action in &plan.actions {
        if action.action_id.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::EmptyActionIdentity);
        }
        if !action_ids.insert(action.action_id.as_str()) {
            diagnostics.push(ExperimentValueDiagnostic::DuplicateAction);
        }
        if action.evidence_ref.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::EmptyActionEvidence);
        }
        match action.eligibility {
            ActionEligibility::HardGateUnknown => {
                diagnostics.push(ExperimentValueDiagnostic::UnknownActionEligibility);
            }
            ActionEligibility::Eligible => {
                match action.current_expected_utility {
                    None => diagnostics.push(ExperimentValueDiagnostic::MissingCurrentUtility),
                    Some(value) if !value.is_finite() => {
                        diagnostics.push(ExperimentValueDiagnostic::NonFiniteCurrentUtility);
                    }
                    Some(_) => {}
                }
            }
            ActionEligibility::HardGateBlocked => {}
        }
    }

    match plan.current_decision_action_id.as_deref() {
        None => diagnostics.push(ExperimentValueDiagnostic::MissingCurrentAction),
        Some(action_id) => match plan.actions.iter().find(|action| action.action_id == action_id) {
            None => diagnostics.push(ExperimentValueDiagnostic::UnknownCurrentAction),
            Some(action) if action.eligibility == ActionEligibility::HardGateBlocked => {
                diagnostics.push(ExperimentValueDiagnostic::CurrentActionBlocked);
            }
            Some(action) if action.eligibility == ActionEligibility::HardGateUnknown => {
                diagnostics.push(ExperimentValueDiagnostic::UnknownActionEligibility);
            }
            Some(_) => {}
        },
    }
    diagnostics
}

fn evaluate_scenario_impact(
    plan: &ExperimentValuePlan,
    diagnostics: &mut Vec<ExperimentValueDiagnostic>,
) -> Option<ScenarioImpactSummary> {
    if plan.scenario_outcomes.is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::MissingScenarioOutcomes);
        return None;
    }
    let current_action = plan.current_decision_action_id.as_deref()?;
    let mut ids = std::collections::BTreeSet::new();
    let mut outcomes = plan.scenario_outcomes.clone();
    outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
    let mut unchanged = 0;
    let mut changed = 0;
    let mut unresolved = 0;
    let mut resolves = 0;
    let mut gate_failures = 0;

    for outcome in &outcomes {
        if outcome.outcome_id.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::MissingScenarioOutcomeIdentity);
        }
        if !ids.insert(outcome.outcome_id.as_str()) {
            diagnostics.push(ExperimentValueDiagnostic::DuplicateScenarioOutcome);
        }
        if outcome.evidence_ref.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::EmptyScenarioOutcomeEvidence);
        }
        match outcome.post_outcome_action_id.as_deref() {
            None => unresolved += 1,
            Some(action_id) => match plan.actions.iter().find(|action| action.action_id == action_id) {
                None => diagnostics.push(ExperimentValueDiagnostic::UnknownPostOutcomeAction),
                Some(action) if action.eligibility == ActionEligibility::HardGateBlocked => {
                    diagnostics.push(ExperimentValueDiagnostic::PostOutcomeActionBlocked);
                }
                Some(action) if action.eligibility == ActionEligibility::HardGateUnknown => {
                    diagnostics.push(ExperimentValueDiagnostic::UnknownActionEligibility);
                }
                Some(_) if action_id == current_action => unchanged += 1,
                Some(_) => changed += 1,
            },
        }
        if plan.current_decision_is_split && outcome.resolves_current_disagreement {
            resolves += 1;
        }
        if outcome.reveals_new_hard_gate_failure {
            gate_failures += 1;
        }
    }

    Some(ScenarioImpactSummary {
        outcome_count: outcomes.len(),
        unchanged_action_count: unchanged,
        changed_action_count: changed,
        unresolved_action_count: unresolved,
        resolves_current_disagreement_count: resolves,
        reveals_hard_gate_failure_count: gate_failures,
        outcomes,
    })
}

fn evaluate_calibrated_evsi(
    plan: &ExperimentValuePlan,
    diagnostics: &mut Vec<ExperimentValueDiagnostic>,
) -> Option<CalibratedEvsiSummary> {
    if plan.evidence_class != ExperimentEvidenceClass::CalibratedProbabilisticModel {
        diagnostics.push(match plan.evidence_class {
            ExperimentEvidenceClass::SyntheticFixture => ExperimentValueDiagnostic::SyntheticEvsiRefused,
            ExperimentEvidenceClass::Assumption => ExperimentValueDiagnostic::AssumptionOnlyEvsiRefused,
            _ => ExperimentValueDiagnostic::UncalibratedEvsiRefused,
        });
    }
    if plan.feasibility == ExperimentFeasibility::NotAvailable {
        diagnostics.push(ExperimentValueDiagnostic::ExperimentInfeasible);
    } else if plan.feasibility == ExperimentFeasibility::Unknown {
        diagnostics.push(ExperimentValueDiagnostic::ExperimentFeasibilityUnknown);
    }

    let probability_model_ref = plan.probability_model_ref.as_deref();
    let calibration_ref = plan.calibration_manifest_ref.as_deref();
    let joint_outcome_ref = plan.joint_outcome_model_ref.as_deref();
    if probability_model_ref.is_none_or(str::is_empty) {
        diagnostics.push(ExperimentValueDiagnostic::MissingProbabilityModel);
    }
    if plan.probability_model_sha256.as_deref().is_none_or(|value| !is_sha256(value)) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidProbabilityModelDigest);
    }
    if calibration_ref.is_none_or(str::is_empty) {
        diagnostics.push(ExperimentValueDiagnostic::MissingCalibrationManifest);
    }
    if plan.calibration_manifest_sha256.as_deref().is_none_or(|value| !is_sha256(value)) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidCalibrationManifestDigest);
    }
    if joint_outcome_ref.is_none_or(str::is_empty) {
        diagnostics.push(ExperimentValueDiagnostic::MissingJointOutcomeModel);
    }
    if plan.joint_outcome_model_sha256.as_deref().is_none_or(|value| !is_sha256(value)) {
        diagnostics.push(ExperimentValueDiagnostic::InvalidJointOutcomeModelDigest);
    }
    if !plan.probability_sum_tolerance.is_finite() || plan.probability_sum_tolerance < 0.0 {
        diagnostics.push(ExperimentValueDiagnostic::InvalidProbabilitySumTolerance);
    }
    if plan.probabilistic_outcomes.is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::MissingProbabilisticOutcomes);
    }
    let Some(cost) = plan.experiment_cost_utility_units else {
        diagnostics.push(ExperimentValueDiagnostic::MissingCostUtility);
        return None;
    };
    if !cost.is_finite() || cost < 0.0 {
        diagnostics.push(ExperimentValueDiagnostic::InvalidCostUtility);
    }
    if plan.cost_utility_basis_ref.as_deref() != Some(plan.utility_basis_ref.as_str()) {
        diagnostics.push(ExperimentValueDiagnostic::CostUtilityBasisMismatch);
    }
    if plan.cost_utility_evidence_ref.as_deref().is_none_or(str::is_empty) {
        diagnostics.push(ExperimentValueDiagnostic::MissingCostEvidence);
    }

    if !diagnostics.is_empty() {
        return None;
    }

    let mut outcome_ids = std::collections::BTreeSet::new();
    let mut outcomes = plan.probabilistic_outcomes.clone();
    outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
    let eligible_actions: Vec<&ExperimentAction> = plan
        .actions
        .iter()
        .filter(|action| action.eligibility == ActionEligibility::Eligible)
        .collect();
    if eligible_actions.is_empty() {
        diagnostics.push(ExperimentValueDiagnostic::MissingCurrentAction);
        return None;
    }

    let current_utilities: Vec<(String, f64)> = eligible_actions
        .iter()
        .filter_map(|action| {
            action.current_expected_utility.map(|value| (action.action_id.clone(), value))
        })
        .collect();
    let Some((current_best_action_id, current_best_expected_utility)) =
        best_action(&current_utilities)
    else {
        diagnostics.push(ExperimentValueDiagnostic::MissingCurrentUtility);
        return None;
    };

    let mut probability_sum = 0.0;
    let mut expected_post_best = 0.0;
    let mut decision_change_probability = 0.0;
    let mut summaries = Vec::new();

    for outcome in &outcomes {
        if outcome.outcome_id.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::EmptyProbabilisticOutcomeIdentity);
        }
        if !outcome_ids.insert(outcome.outcome_id.as_str()) {
            diagnostics.push(ExperimentValueDiagnostic::DuplicateProbabilisticOutcome);
        }
        if outcome.probability_evidence_ref.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::MissingProbabilityEvidence);
        }
        if outcome.utility_evidence_ref.trim().is_empty() {
            diagnostics.push(ExperimentValueDiagnostic::MissingUtilityEvidence);
        }
        if !outcome.probability.is_finite() || outcome.probability <= 0.0 || outcome.probability > 1.0 {
            diagnostics.push(ExperimentValueDiagnostic::InvalidOutcomeProbability);
        }
        probability_sum += outcome.probability;

        let mut conditional_ids = std::collections::BTreeSet::new();
        let mut utilities = Vec::new();
        for conditional in &outcome.conditional_utilities {
            if conditional.evidence_ref.trim().is_empty() {
                diagnostics.push(ExperimentValueDiagnostic::MissingConditionalUtilityEvidence);
            }
            if !conditional.expected_utility.is_finite() {
                diagnostics.push(ExperimentValueDiagnostic::NonFiniteConditionalUtility);
            }
            if !conditional_ids.insert(conditional.action_id.as_str()) {
                diagnostics.push(ExperimentValueDiagnostic::DuplicateConditionalUtility);
            }
            if !eligible_actions.iter().any(|action| action.action_id == conditional.action_id) {
                diagnostics.push(ExperimentValueDiagnostic::UnknownConditionalUtilityAction);
            }
            utilities.push((conditional.action_id.clone(), conditional.expected_utility));
        }
        for action in &eligible_actions {
            if !conditional_ids.contains(action.action_id.as_str()) {
                diagnostics.push(ExperimentValueDiagnostic::MissingConditionalUtility);
            }
        }
        let Some((best_id, best_utility)) = best_action(&utilities) else {
            diagnostics.push(ExperimentValueDiagnostic::MissingConditionalUtility);
            continue;
        };
        expected_post_best += outcome.probability * best_utility;
        if best_id != current_best_action_id {
            decision_change_probability += outcome.probability;
        }
        summaries.push(EvsiOutcomeSummary {
            outcome_id: outcome.outcome_id.clone(),
            probability: outcome.probability,
            best_action_id: best_id,
            best_conditional_expected_utility: best_utility,
        });
    }

    if (probability_sum - 1.0).abs() > plan.probability_sum_tolerance {
        diagnostics.push(ExperimentValueDiagnostic::ProbabilitySumMismatch);
    }
    if !diagnostics.is_empty() {
        return None;
    }

    let gross_evsi = expected_post_best - current_best_expected_utility;
    Some(CalibratedEvsiSummary {
        current_best_action_id,
        current_best_expected_utility,
        expected_post_experiment_best_utility: expected_post_best,
        gross_evsi_utility_units: gross_evsi,
        experiment_cost_utility_units: cost,
        net_evsi_utility_units: gross_evsi - cost,
        decision_change_probability,
        outcomes: summaries,
    })
}

/// Highest expected utility wins. Ties are resolved by lexicographically
/// smallest action ID, so reports are stable under input reordering.
fn best_action(values: &[(String, f64)]) -> Option<(String, f64)> {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));
    let mut best: Option<(String, f64)> = None;
    for (action_id, utility) in ordered {
        match &best {
            None => best = Some((action_id, utility)),
            Some((_, best_utility)) if utility > *best_utility => best = Some((action_id, utility)),
            _ => {}
        }
    }
    best
}

fn collect_evidence_refs(plan: &ExperimentValuePlan) -> Vec<String> {
    let mut refs = vec![
        plan.experiment_protocol_ref.clone(),
        plan.decision_profile_ref.clone(),
        plan.decision_envelope_ref.clone(),
    ];
    refs.extend(plan.actions.iter().map(|action| action.evidence_ref.clone()));
    refs.extend(plan.scenario_outcomes.iter().map(|outcome| outcome.evidence_ref.clone()));
    refs.extend(plan.probabilistic_outcomes.iter().flat_map(|outcome| {
        vec![
            outcome.probability_evidence_ref.clone(),
            outcome.utility_evidence_ref.clone(),
        ]
    }));
    if let Some(reference) = &plan.probability_model_ref {
        refs.push(reference.clone());
    }
    if let Some(reference) = &plan.calibration_manifest_ref {
        refs.push(reference.clone());
    }
    if let Some(reference) = &plan.joint_outcome_model_ref {
        refs.push(reference.clone());
    }
    if let Some(reference) = &plan.cost_utility_evidence_ref {
        refs.push(reference.clone());
    }
    refs.sort();
    refs.dedup();
    refs
}

fn invalid_report(
    plan: &ExperimentValuePlan,
    diagnostics: Vec<ExperimentValueDiagnostic>,
    evidence_refs: Vec<String>,
) -> ExperimentValueReport {
    ExperimentValueReport {
        contract_version: EXPERIMENT_VALUE_VERSION,
        experiment_id: plan.experiment_id.clone(),
        experiment_revision: plan.experiment_revision.clone(),
        mode: plan.mode,
        disposition: ExperimentValueDisposition::NotEvaluable,
        feasibility: plan.feasibility,
        duration_hours: plan.duration_hours,
        scenario_impact: None,
        calibrated_evsi: None,
        diagnostics,
        evidence_refs,
        claim_ceiling: "unweighted finite experiment-outcome decision-impact report only",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn action(id: &str, current_utility: Option<f64>, eligibility: ActionEligibility) -> ExperimentAction {
        ExperimentAction {
            action_id: id.to_string(),
            eligibility,
            current_expected_utility: current_utility,
            evidence_ref: format!("action-evidence-{id}"),
        }
    }

    fn base_plan() -> ExperimentValuePlan {
        ExperimentValuePlan {
            experiment_id: "experiment-blanket-coupon-irradiation".to_string(),
            experiment_revision: "protocol-r3".to_string(),
            experiment_protocol_ref: "protocol-ref".to_string(),
            experiment_protocol_sha256: sha('a'),
            decision_profile_ref: "decision-profile-ref".to_string(),
            decision_profile_sha256: sha('b'),
            decision_envelope_ref: "decision-envelope-ref".to_string(),
            decision_envelope_sha256: sha('c'),
            utility_basis_ref: "utility-usd-equivalent-2026-v1".to_string(),
            utility_basis_sha256: sha('d'),
            evidence_class: ExperimentEvidenceClass::SyntheticFixture,
            feasibility: ExperimentFeasibility::Available,
            duration_hours: 720.0,
            experiment_cost_utility_units: None,
            cost_utility_basis_ref: None,
            cost_utility_evidence_ref: None,
            current_decision_action_id: Some("continue-design-a".to_string()),
            current_decision_is_split: true,
            actions: vec![
                action("continue-design-a", Some(10.0), ActionEligibility::Eligible),
                action("switch-design-b", Some(8.0), ActionEligibility::Eligible),
                action("blocked-high-score", None, ActionEligibility::HardGateBlocked),
            ],
            mode: ExperimentMode::ScenarioImpact,
            probability_model_ref: None,
            probability_model_sha256: None,
            calibration_manifest_ref: None,
            calibration_manifest_sha256: None,
            joint_outcome_model_ref: None,
            joint_outcome_model_sha256: None,
            probability_sum_tolerance: 1e-9,
            scenario_outcomes: vec![
                ScenarioImpactOutcome {
                    outcome_id: "low-swelling".to_string(),
                    post_outcome_action_id: Some("continue-design-a".to_string()),
                    resolves_current_disagreement: true,
                    reveals_new_hard_gate_failure: false,
                    evidence_ref: "scenario-low-swelling".to_string(),
                },
                ScenarioImpactOutcome {
                    outcome_id: "high-swelling".to_string(),
                    post_outcome_action_id: Some("switch-design-b".to_string()),
                    resolves_current_disagreement: true,
                    reveals_new_hard_gate_failure: true,
                    evidence_ref: "scenario-high-swelling".to_string(),
                },
                ScenarioImpactOutcome {
                    outcome_id: "ambiguous-swelling".to_string(),
                    post_outcome_action_id: None,
                    resolves_current_disagreement: false,
                    reveals_new_hard_gate_failure: false,
                    evidence_ref: "scenario-ambiguous-swelling".to_string(),
                },
            ],
            probabilistic_outcomes: Vec::new(),
        }
    }

    fn calibrated_plan() -> ExperimentValuePlan {
        let mut plan = base_plan();
        plan.mode = ExperimentMode::CalibratedEvsi;
        plan.evidence_class = ExperimentEvidenceClass::CalibratedProbabilisticModel;
        plan.cost_utility_basis_ref = Some(plan.utility_basis_ref.clone());
        plan.experiment_cost_utility_units = Some(1.0);
        plan.cost_utility_evidence_ref = Some("cost-utility-conversion-evidence".to_string());
        plan.probability_model_ref = Some("joint-outcome-probability-model".to_string());
        plan.probability_model_sha256 = Some(sha('e'));
        plan.calibration_manifest_ref = Some("calibration-manifest".to_string());
        plan.calibration_manifest_sha256 = Some(sha('f'));
        plan.joint_outcome_model_ref = Some("joint-outcome-model".to_string());
        plan.joint_outcome_model_sha256 = Some(sha('g'));
        plan.scenario_outcomes.clear();
        plan.probabilistic_outcomes = vec![
            ProbabilisticExperimentOutcome {
                outcome_id: "observation-low".to_string(),
                probability: 0.4,
                conditional_utilities: vec![
                    ConditionalActionUtility {
                        action_id: "continue-design-a".to_string(),
                        expected_utility: 11.0,
                        evidence_ref: "utility-low-a".to_string(),
                    },
                    ConditionalActionUtility {
                        action_id: "switch-design-b".to_string(),
                        expected_utility: 8.0,
                        evidence_ref: "utility-low-b".to_string(),
                    },
                ],
                probability_evidence_ref: "probability-low".to_string(),
                utility_evidence_ref: "utility-model-low".to_string(),
            },
            ProbabilisticExperimentOutcome {
                outcome_id: "observation-high".to_string(),
                probability: 0.6,
                conditional_utilities: vec![
                    ConditionalActionUtility {
                        action_id: "continue-design-a".to_string(),
                        expected_utility: 6.0,
                        evidence_ref: "utility-high-a".to_string(),
                    },
                    ConditionalActionUtility {
                        action_id: "switch-design-b".to_string(),
                        expected_utility: 12.0,
                        evidence_ref: "utility-high-b".to_string(),
                    },
                ],
                probability_evidence_ref: "probability-high".to_string(),
                utility_evidence_ref: "utility-model-high".to_string(),
            },
        ];
        plan
    }

    #[test]
    fn scenario_impact_mode_counts_outcomes_without_probabilities() {
        let report = evaluate_experiment_value(&base_plan());
        assert_eq!(report.disposition, ExperimentValueDisposition::ScenarioImpactOnly);
        let impact = report.scenario_impact.unwrap();
        assert_eq!(impact.outcome_count, 3);
        assert_eq!(impact.unchanged_action_count, 1);
        assert_eq!(impact.changed_action_count, 1);
        assert_eq!(impact.unresolved_action_count, 1);
        assert_eq!(impact.resolves_current_disagreement_count, 2);
        assert_eq!(impact.reveals_hard_gate_failure_count, 1);
        assert_eq!(report.calibrated_evsi, None);
    }

    #[test]
    fn scenario_impact_mode_does_not_invent_probability_weighting() {
        let report = evaluate_experiment_value(&base_plan());
        assert!(report.calibrated_evsi.is_none());
        assert!(report.scenario_impact.is_some());
    }

    #[test]
    fn scenario_mode_rejects_probability_fields() {
        let mut plan = base_plan();
        plan.probability_model_ref = Some("uncalibrated-probability".to_string());
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::InconsistentModeInputs));
    }

    #[test]
    fn scenario_impact_cannot_select_a_hard_gate_blocked_action() {
        let mut plan = base_plan();
        plan.scenario_outcomes[0].post_outcome_action_id = Some("blocked-high-score".to_string());
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::PostOutcomeActionBlocked));
    }

    #[test]
    fn calibrated_evsi_uses_probability_weighted_post_experiment_optima() {
        let report = evaluate_experiment_value(&calibrated_plan());
        assert_eq!(report.disposition, ExperimentValueDisposition::CalibratedEvsiComputed);
        let summary = report.calibrated_evsi.unwrap();
        assert_eq!(summary.current_best_action_id, "continue-design-a");
        assert!((summary.expected_post_experiment_best_utility - 11.6).abs() < 1e-12);
        assert!((summary.gross_evsi_utility_units - 1.6).abs() < 1e-12);
        assert!((summary.net_evsi_utility_units - 0.6).abs() < 1e-12);
        assert!((summary.decision_change_probability - 0.6).abs() < 1e-12);
    }

    #[test]
    fn negative_net_evsi_is_preserved() {
        let mut plan = calibrated_plan();
        plan.experiment_cost_utility_units = Some(5.0);
        let report = evaluate_experiment_value(&plan);
        assert!((report.calibrated_evsi.unwrap().net_evsi_utility_units + 3.4).abs() < 1e-12);
    }

    #[test]
    fn probability_sum_mismatch_refuses_evsi() {
        let mut plan = calibrated_plan();
        plan.probabilistic_outcomes[1].probability = 0.5;
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::ProbabilitySumMismatch));
    }

    #[test]
    fn missing_conditional_utility_refuses_evsi() {
        let mut plan = calibrated_plan();
        plan.probabilistic_outcomes[0].conditional_utilities.pop();
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::MissingConditionalUtility));
    }

    #[test]
    fn unknown_and_hard_blocked_actions_are_not_selected_or_silently_admitted() {
        let mut plan = calibrated_plan();
        plan.actions.push(action("unknown-action", None, ActionEligibility::HardGateUnknown));
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::UnknownActionEligibility));
    }

    #[test]
    fn probability_model_requires_calibration_class() {
        let mut plan = calibrated_plan();
        plan.evidence_class = ExperimentEvidenceClass::ModelledUncalibrated;
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::UncalibratedEvsiRefused));
    }

    #[test]
    fn synthetic_or_assumption_evidence_cannot_produce_evsi() {
        let mut plan = calibrated_plan();
        plan.evidence_class = ExperimentEvidenceClass::SyntheticFixture;
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::SyntheticEvsiRefused));
        plan.evidence_class = ExperimentEvidenceClass::Assumption;
        let report = evaluate_experiment_value(&plan);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::AssumptionOnlyEvsiRefused));
    }

    #[test]
    fn cost_utility_must_share_exact_utility_basis() {
        let mut plan = calibrated_plan();
        plan.cost_utility_basis_ref = Some("different-utility-basis".to_string());
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::CostUtilityBasisMismatch));
    }

    #[test]
    fn infeasible_or_unknown_experiment_does_not_emit_evsi() {
        let mut plan = calibrated_plan();
        plan.feasibility = ExperimentFeasibility::NotAvailable;
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::ExperimentInfeasible));
        plan.feasibility = ExperimentFeasibility::Unknown;
        let report = evaluate_experiment_value(&plan);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::ExperimentFeasibilityUnknown));
    }

    #[test]
    fn shared_and_complete_utility_inputs_are_order_invariant() {
        let first = evaluate_experiment_value(&calibrated_plan());
        let mut reversed = calibrated_plan();
        reversed.actions.reverse();
        reversed.probabilistic_outcomes.reverse();
        for outcome in &mut reversed.probabilistic_outcomes {
            outcome.conditional_utilities.reverse();
        }
        let second = evaluate_experiment_value(&reversed);
        assert_eq!(first, second);
    }

    #[test]
    fn scenario_outcomes_reject_duplicate_identity_and_missing_evidence() {
        let mut plan = base_plan();
        plan.scenario_outcomes[1].outcome_id = plan.scenario_outcomes[0].outcome_id.clone();
        plan.scenario_outcomes[2].evidence_ref.clear();
        let report = evaluate_experiment_value(&plan);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::DuplicateScenarioOutcome));
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::EmptyScenarioOutcomeEvidence));
    }

    #[test]
    fn unknown_post_outcome_action_is_refused() {
        let mut plan = base_plan();
        plan.scenario_outcomes[0].post_outcome_action_id = Some("not-an-action".to_string());
        let report = evaluate_experiment_value(&plan);
        assert_eq!(report.disposition, ExperimentValueDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::UnknownPostOutcomeAction));
    }

    #[test]
    fn empty_joint_probability_and_calibration_receipts_are_refused() {
        let mut plan = calibrated_plan();
        plan.probability_model_ref = None;
        plan.calibration_manifest_ref = None;
        plan.joint_outcome_model_ref = None;
        let report = evaluate_experiment_value(&plan);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::MissingProbabilityModel));
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::MissingCalibrationManifest));
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::MissingJointOutcomeModel));
    }

    #[test]
    fn experiment_duration_and_protocol_identity_are_required() {
        let mut plan = base_plan();
        plan.duration_hours = 0.0;
        plan.experiment_protocol_sha256 = "bad".to_string();
        let report = evaluate_experiment_value(&plan);
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::InvalidDuration));
        assert!(report.diagnostics.contains(&ExperimentValueDiagnostic::InvalidProtocolDigest));
    }
}
