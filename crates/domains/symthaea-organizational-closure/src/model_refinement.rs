//! Open-world model adequacy and held-out hypothesis refinement for AOBench.
//!
//! The evaluator distinguishes a hypothesis class that is still consistent
//! with qualified observations from one falsified by the complete trace.
//! A proposed extension must explain discovery observations and predict
//! sealed validation probes from an evaluator-only world oracle.
//!
//! This deterministic finite-model primitive does not establish unrestricted
//! scientific model discovery, stochastic power or physical-world truth.

use super::experiment_design::{
    ExperimentBudget, ExperimentEvidenceStatus, ExperimentTrial, HypothesisId, MechanismHypothesis,
    ProbeId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A probe descriptor. It contains no outcome or oracle information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefinementProbe {
    pub id: ProbeId,
    pub cost_units: u64,
    pub risk_units: u64,
}

impl RefinementProbe {
    pub fn new(id: impl Into<ProbeId>, cost_units: u64, risk_units: u64) -> Self {
        Self {
            id: id.into(),
            cost_units,
            risk_units,
        }
    }
}

/// One supplied mechanism and its frozen deterministic predictions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateMechanismModel {
    pub hypothesis: MechanismHypothesis,
    pub predicted_outcomes: BTreeMap<ProbeId, String>,
}

impl CandidateMechanismModel {
    pub fn new(
        hypothesis: MechanismHypothesis,
        predicted_outcomes: BTreeMap<ProbeId, String>,
    ) -> Self {
        Self {
            hypothesis,
            predicted_outcomes,
        }
    }
}

/// Public task information. The agent receives this structure and the
/// discovery observations, but never receives ModelRefinementOracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdequacyScenario {
    pub known_models: Vec<CandidateMechanismModel>,
    pub probes: Vec<RefinementProbe>,
    /// These probe IDs are declared before evaluation and are not executed
    /// during discovery. Their true outcomes remain evaluator-only.
    pub validation_probe_ids: BTreeSet<ProbeId>,
    pub budget: ExperimentBudget,
}

/// Evaluator-only truth, supplied to the harness through a separate channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRefinementOracle {
    pub actual_target_class: String,
    pub actual_outcomes: BTreeMap<ProbeId, String>,
}

impl ModelRefinementOracle {
    pub fn new(
        actual_target_class: impl Into<String>,
        actual_outcomes: BTreeMap<ProbeId, String>,
    ) -> Self {
        Self {
            actual_target_class: actual_target_class.into(),
            actual_outcomes,
        }
    }
}

/// A new falsifiable model. Predictions must cover the complete declared
/// probe universe; sealed probes are checked against the hidden oracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRefinementProposal {
    pub hypothesis: MechanismHypothesis,
    pub predicted_outcomes: BTreeMap<ProbeId, String>,
}

impl ModelRefinementProposal {
    pub fn new(
        hypothesis: MechanismHypothesis,
        predicted_outcomes: BTreeMap<ProbeId, String>,
    ) -> Self {
        Self {
            hypothesis,
            predicted_outcomes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelAdequacyDecision {
    RetainExistingModel { rationale: String },
    ExpandModelClass { proposal: ModelRefinementProposal },
    DeferAsUnresolved { rationale: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdequacyTrace {
    pub discovery_trials: Vec<ExperimentTrial>,
    pub decision: ModelAdequacyDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAdequacyStatus {
    NotFalsified,
    Falsified,
    Unresolved,
    RefinedAndValidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRefinementQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRefinementFindingCode {
    EmptyKnownModelSet,
    EmptyProbeSet,
    DuplicateHypothesisId,
    EmptyHypothesisId,
    EmptyTargetClass,
    ZeroPriorWeight,
    DuplicateProbeId,
    EmptyProbeId,
    DuplicateValidationProbe,
    UnknownValidationProbe,
    EmptyValidationSet,
    MissingModelPrediction,
    UnknownModelPrediction,
    EmptyModelPrediction,
    MissingOraclePrediction,
    EmptyOraclePrediction,
    UnknownOraclePrediction,
    UnknownObservedProbe,
    EmptyObservedOutcome,
    DeliveryRejected,
    IsolationRejected,
    ObservationRejected,
    EvidenceUnavailable,
    CostBudgetExceeded,
    RiskBudgetExceeded,
    TrialBudgetExceeded,
    ObservationContradictsOracle,
    NoExistingModelFitsTrace,
    ExistingModelStillFitsTrace,
    RefinementIdAlreadyExists,
    EmptyProposalPrediction,
    MissingProposalPrediction,
    UnknownProposalPrediction,
    ProposalDoesNotExplainDiscovery,
    ProposalTargetClassMismatch,
    ValidationPredictionMismatch,
    InvalidProposalCoverage,
    DeferredWithoutEvidence,
    RetainedFalsifiedModel,
    ExpandedAdequateModelClass,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRefinementFinding {
    pub code: ModelRefinementFindingCode,
    pub trial_index: Option<usize>,
    pub probe_id: Option<ProbeId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRefinementReport {
    pub qualification: ModelRefinementQualification,
    pub model_adequacy: ModelAdequacyStatus,
    pub initial_consistent_model_ids: Vec<HypothesisId>,
    pub discovery_trial_count: usize,
    pub qualified_discovery_trial_count: usize,
    pub total_cost_units: u64,
    pub total_risk_units: u64,
    pub validation_probe_count: usize,
    pub validation_prediction_count: usize,
    pub validation_correct_count: usize,
    pub validation_accuracy: Option<f64>,
    pub model_inadequacy_detected: bool,
    pub safe_defer: bool,
    pub refinement_accepted: bool,
    pub findings: Vec<ModelRefinementFinding>,
}

impl ModelRefinementReport {
    pub fn validates_new_model(&self) -> bool {
        self.qualification == ModelRefinementQualification::Qualified
            && self.model_adequacy == ModelAdequacyStatus::RefinedAndValidated
            && self.refinement_accepted
            && self.validation_accuracy == Some(1.0)
            && self.findings.is_empty()
    }

    pub fn correctly_retains_model_class(&self) -> bool {
        self.qualification == ModelRefinementQualification::Qualified
            && self.model_adequacy == ModelAdequacyStatus::NotFalsified
            && !self.model_inadequacy_detected
            && self.findings.is_empty()
    }
}

fn push_finding(
    findings: &mut Vec<ModelRefinementFinding>,
    code: ModelRefinementFindingCode,
    trial_index: Option<usize>,
    probe_id: Option<ProbeId>,
    detail: impl Into<String>,
) {
    findings.push(ModelRefinementFinding {
        code,
        trial_index,
        probe_id,
        detail: detail.into(),
    });
}

struct ModelRefinementReportInput<'a> {
    qualification: ModelRefinementQualification,
    model_adequacy: ModelAdequacyStatus,
    consistent_ids: Vec<HypothesisId>,
    trace: &'a ModelAdequacyTrace,
    qualified_trials: usize,
    cost: u64,
    risk: u64,
    validation_count: usize,
    validation_predictions: usize,
    validation_correct: usize,
    model_inadequacy_detected: bool,
    safe_defer: bool,
    refinement_accepted: bool,
    findings: Vec<ModelRefinementFinding>,
}

fn report(input: ModelRefinementReportInput<'_>) -> ModelRefinementReport {
    ModelRefinementReport {
        qualification: input.qualification,
        model_adequacy: input.model_adequacy,
        initial_consistent_model_ids: input.consistent_ids,
        discovery_trial_count: input.trace.discovery_trials.len(),
        qualified_discovery_trial_count: input.qualified_trials,
        total_cost_units: input.cost,
        total_risk_units: input.risk,
        validation_probe_count: input.validation_count,
        validation_prediction_count: input.validation_predictions,
        validation_correct_count: input.validation_correct,
        validation_accuracy: if input.validation_predictions == 0 {
            None
        } else {
            Some(input.validation_correct as f64 / input.validation_predictions as f64)
        },
        model_inadequacy_detected: input.model_inadequacy_detected,
        safe_defer: input.safe_defer,
        refinement_accepted: input.refinement_accepted,
        findings: input.findings,
    }
}
fn invalid_scenario(
    trace: &ModelAdequacyTrace,
    findings: Vec<ModelRefinementFinding>,
    consistent_ids: Vec<HypothesisId>,
) -> ModelRefinementReport {
    report(ModelRefinementReportInput {
        qualification: ModelRefinementQualification::Inconclusive,
        model_adequacy: ModelAdequacyStatus::Unresolved,
        consistent_ids,
        trace,
        qualified_trials: 0,
        cost: 0,
        risk: 0,
        validation_count: 0,
        validation_predictions: 0,
        validation_correct: 0,
        model_inadequacy_detected: false,
        safe_defer: false,
        refinement_accepted: false,
        findings,
    })
}

/// Evaluate model-class adequacy and any proposed extension.
///
/// All discovery outcomes must have independently verified delivery,
/// isolation and observation evidence. Existing models are tested against the
/// entire qualified trace. A refinement must explain every discovery outcome
/// and correctly predict every sealed validation probe from the hidden oracle.
pub fn evaluate_model_refinement(
    scenario: &ModelAdequacyScenario,
    oracle: &ModelRefinementOracle,
    trace: &ModelAdequacyTrace,
) -> ModelRefinementReport {
    let mut findings = Vec::new();

    if scenario.known_models.is_empty() {
        push_finding(
            &mut findings,
            ModelRefinementFindingCode::EmptyKnownModelSet,
            None,
            None,
            "no baseline mechanism models were declared",
        );
    }
    if scenario.probes.is_empty() {
        push_finding(
            &mut findings,
            ModelRefinementFindingCode::EmptyProbeSet,
            None,
            None,
            "no experiment probes were declared",
        );
    }

    let mut probes = BTreeMap::new();
    for probe in &scenario.probes {
        if probe.id.0.trim().is_empty() {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::EmptyProbeId,
                None,
                Some(probe.id.clone()),
                "probe ID is empty",
            );
        }
        if probes.insert(probe.id.clone(), probe).is_some() {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::DuplicateProbeId,
                None,
                Some(probe.id.clone()),
                "probe ID is duplicated",
            );
        }
    }

    if scenario.validation_probe_ids.is_empty() {
        push_finding(
            &mut findings,
            ModelRefinementFindingCode::EmptyValidationSet,
            None,
            None,
            "at least one sealed validation probe is required",
        );
    }
    for id in &scenario.validation_probe_ids {
        if !probes.contains_key(id) {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::UnknownValidationProbe,
                None,
                Some(id.clone()),
                "validation set references an undeclared probe",
            );
        }
    }
    for (index, id) in scenario.validation_probe_ids.iter().enumerate() {
        if scenario
            .validation_probe_ids
            .iter()
            .take(index)
            .any(|previous| previous == id)
        {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::DuplicateValidationProbe,
                None,
                Some(id.clone()),
                "validation probe ID is duplicated",
            );
        }
    }

    let mut models = BTreeMap::new();
    for model in &scenario.known_models {
        let id = &model.hypothesis.id;
        if id.0.trim().is_empty() {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::EmptyHypothesisId,
                None,
                None,
                "known model hypothesis ID is empty",
            );
        }
        if model.hypothesis.target_class.trim().is_empty() {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::EmptyTargetClass,
                None,
                None,
                "known model target-equivalence class is empty",
            );
        }
        if model.hypothesis.prior_weight == 0 {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::ZeroPriorWeight,
                None,
                None,
                "known model prior weight must be positive",
            );
        }
        if models.insert(id.clone(), model).is_some() {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::DuplicateHypothesisId,
                None,
                None,
                "known model hypothesis ID is duplicated",
            );
        }

        for probe_id in probes.keys() {
            match model.predicted_outcomes.get(probe_id) {
                None => push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::MissingModelPrediction,
                    None,
                    Some(probe_id.clone()),
                    format!("known model {} has no probe prediction", id.0),
                ),
                Some(value) if value.trim().is_empty() => push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::EmptyModelPrediction,
                    None,
                    Some(probe_id.clone()),
                    format!("known model {} has an empty probe prediction", id.0),
                ),
                Some(_) => {}
            }
        }
        for probe_id in model.predicted_outcomes.keys() {
            if !probes.contains_key(probe_id) {
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::UnknownModelPrediction,
                    None,
                    Some(probe_id.clone()),
                    format!("known model {} predicts an unknown probe", id.0),
                );
            }
        }
    }

    if oracle.actual_target_class.trim().is_empty() {
        push_finding(
            &mut findings,
            ModelRefinementFindingCode::EmptyTargetClass,
            None,
            None,
            "evaluator oracle target-equivalence class is empty",
        );
    }
    for probe_id in probes.keys() {
        match oracle.actual_outcomes.get(probe_id) {
            None => push_finding(
                &mut findings,
                ModelRefinementFindingCode::MissingOraclePrediction,
                None,
                Some(probe_id.clone()),
                "evaluator oracle has no outcome for a declared probe",
            ),
            Some(value) if value.trim().is_empty() => push_finding(
                &mut findings,
                ModelRefinementFindingCode::EmptyOraclePrediction,
                None,
                Some(probe_id.clone()),
                "evaluator oracle contains an empty outcome",
            ),
            Some(_) => {}
        }
    }
    for probe_id in oracle.actual_outcomes.keys() {
        if !probes.contains_key(probe_id) {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::UnknownOraclePrediction,
                None,
                Some(probe_id.clone()),
                "evaluator oracle contains an undeclared probe",
            );
        }
    }

    if !findings.is_empty() {
        return invalid_scenario(trace, findings, models.keys().cloned().collect());
    }

    let mut total_cost = 0_u64;
    let mut total_risk = 0_u64;
    let mut qualified_trials = Vec::new();
    let mut unresolved_evidence = false;

    for (index, trial) in trace.discovery_trials.iter().enumerate() {
        let trial_index = index + 1;
        let Some(probe) = probes.get(&trial.probe_id) else {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::UnknownObservedProbe,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "discovery trace references an undeclared probe",
            );
            continue;
        };
        if scenario.validation_probe_ids.contains(&trial.probe_id) {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::UnknownObservedProbe,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "sealed validation probes cannot be used during discovery",
            );
            continue;
        }
        if trial.observed_outcome.trim().is_empty() {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::EmptyObservedOutcome,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "observed outcome is empty",
            );
        }

        let next_cost = total_cost.checked_add(probe.cost_units);
        let next_risk = total_risk.checked_add(probe.risk_units);
        if next_cost.is_none_or(|value| value > scenario.budget.max_cost_units) {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::CostBudgetExceeded,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "discovery trace exceeds the cost budget",
            );
        }
        if next_risk.is_none_or(|value| value > scenario.budget.max_risk_units) {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::RiskBudgetExceeded,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "discovery trace exceeds the risk budget",
            );
        }
        total_cost = next_cost.unwrap_or(u64::MAX);
        total_risk = next_risk.unwrap_or(u64::MAX);
        if trial_index > scenario.budget.max_trials {
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::TrialBudgetExceeded,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "discovery trace exceeds the trial budget",
            );
        }

        let mut qualified = true;
        for (label, status) in [
            ("delivery", trial.delivery),
            ("isolation", trial.isolation),
            ("observation", trial.observation),
        ] {
            match status {
                ExperimentEvidenceStatus::Verified => {}
                ExperimentEvidenceStatus::Rejected => {
                    qualified = false;
                    unresolved_evidence = true;
                    let code = match label {
                        "delivery" => ModelRefinementFindingCode::DeliveryRejected,
                        "isolation" => ModelRefinementFindingCode::IsolationRejected,
                        _ => ModelRefinementFindingCode::ObservationRejected,
                    };
                    push_finding(
                        &mut findings,
                        code,
                        Some(trial_index),
                        Some(trial.probe_id.clone()),
                        format!("{label} evidence was rejected; model adequacy cannot be inferred"),
                    );
                }
                ExperimentEvidenceStatus::Unavailable => {
                    qualified = false;
                    unresolved_evidence = true;
                    push_finding(
                        &mut findings,
                        ModelRefinementFindingCode::EvidenceUnavailable,
                        Some(trial_index),
                        Some(trial.probe_id.clone()),
                        format!(
                            "{label} evidence is unavailable; model adequacy cannot be inferred"
                        ),
                    );
                }
            }
        }

        if qualified && trial.observed_outcome != oracle.actual_outcomes[&trial.probe_id] {
            unresolved_evidence = true;
            qualified = false;
            push_finding(
                &mut findings,
                ModelRefinementFindingCode::ObservationContradictsOracle,
                Some(trial_index),
                Some(trial.probe_id.clone()),
                "qualified observation differs from the independent world oracle; do not attribute this to model-class failure",
            );
        }
        if qualified {
            qualified_trials.push(trial);
        }
    }

    if findings.iter().any(|finding| {
        matches!(
            finding.code,
            ModelRefinementFindingCode::UnknownObservedProbe
                | ModelRefinementFindingCode::EmptyObservedOutcome
                | ModelRefinementFindingCode::CostBudgetExceeded
                | ModelRefinementFindingCode::RiskBudgetExceeded
                | ModelRefinementFindingCode::TrialBudgetExceeded
        )
    }) {
        return report(ModelRefinementReportInput {
            qualification: ModelRefinementQualification::NotQualified,
            model_adequacy: ModelAdequacyStatus::Unresolved,
            consistent_ids: Vec::new(),
            trace,
            qualified_trials: qualified_trials.len(),
            cost: total_cost,
            risk: total_risk,
            validation_count: scenario.validation_probe_ids.len(),
            validation_predictions: 0,
            validation_correct: 0,
            model_inadequacy_detected: false,
            safe_defer: false,
            refinement_accepted: false,
            findings,
        });
    }

    if unresolved_evidence {
        return report(ModelRefinementReportInput {
            qualification: ModelRefinementQualification::Inconclusive,
            model_adequacy: ModelAdequacyStatus::Unresolved,
            consistent_ids: Vec::new(),
            trace,
            qualified_trials: qualified_trials.len(),
            cost: total_cost,
            risk: total_risk,
            validation_count: scenario.validation_probe_ids.len(),
            validation_predictions: 0,
            validation_correct: 0,
            model_inadequacy_detected: false,
            safe_defer: matches!(
                trace.decision,
                ModelAdequacyDecision::DeferAsUnresolved { .. }
            ),
            refinement_accepted: false,
            findings,
        });
    }

    let consistent_models: Vec<&CandidateMechanismModel> = models
        .values()
        .copied()
        .filter(|model| {
            qualified_trials.iter().all(|trial| {
                model
                    .predicted_outcomes
                    .get(&trial.probe_id)
                    .is_some_and(|prediction| prediction == &trial.observed_outcome)
            })
        })
        .collect();
    let consistent_ids = consistent_models
        .iter()
        .map(|model| model.hypothesis.id.clone())
        .collect::<Vec<_>>();
    let model_inadequacy_detected = consistent_models.is_empty();
    match &trace.decision {
        ModelAdequacyDecision::RetainExistingModel { rationale: _ } => {
            if model_inadequacy_detected {
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::RetainedFalsifiedModel,
                    None,
                    None,
                    "agent retained the known model class although no model explains the trace",
                );
                report(ModelRefinementReportInput {
                    qualification: ModelRefinementQualification::NotQualified,
                    model_adequacy: ModelAdequacyStatus::Falsified,
                    consistent_ids,
                    trace,
                    qualified_trials: qualified_trials.len(),
                    cost: total_cost,
                    risk: total_risk,
                    validation_count: scenario.validation_probe_ids.len(),
                    validation_predictions: 0,
                    validation_correct: 0,
                    model_inadequacy_detected: true,
                    safe_defer: false,
                    refinement_accepted: false,
                    findings,
                })
            } else {
                findings.retain(|finding| {
                    finding.code != ModelRefinementFindingCode::ExistingModelStillFitsTrace
                });
                report(ModelRefinementReportInput {
                    qualification: ModelRefinementQualification::Qualified,
                    model_adequacy: ModelAdequacyStatus::NotFalsified,
                    consistent_ids,
                    trace,
                    qualified_trials: qualified_trials.len(),
                    cost: total_cost,
                    risk: total_risk,
                    validation_count: scenario.validation_probe_ids.len(),
                    validation_predictions: 0,
                    validation_correct: 0,
                    model_inadequacy_detected: false,
                    safe_defer: false,
                    refinement_accepted: false,
                    findings,
                })
            }
        }
        ModelAdequacyDecision::DeferAsUnresolved { rationale: _ } => {
            if model_inadequacy_detected {
                findings.retain(|finding| {
                    finding.code != ModelRefinementFindingCode::NoExistingModelFitsTrace
                });
                report(ModelRefinementReportInput {
                    qualification: ModelRefinementQualification::Inconclusive,
                    model_adequacy: ModelAdequacyStatus::Falsified,
                    consistent_ids,
                    trace,
                    qualified_trials: qualified_trials.len(),
                    cost: total_cost,
                    risk: total_risk,
                    validation_count: scenario.validation_probe_ids.len(),
                    validation_predictions: 0,
                    validation_correct: 0,
                    model_inadequacy_detected: true,
                    safe_defer: true,
                    refinement_accepted: false,
                    findings,
                })
            } else {
                findings.retain(|finding| {
                    finding.code != ModelRefinementFindingCode::ExistingModelStillFitsTrace
                });
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::DeferredWithoutEvidence,
                    None,
                    None,
                    "agent deferred despite at least one model explaining the complete discovery trace",
                );
                report(ModelRefinementReportInput {
                    qualification: ModelRefinementQualification::NotQualified,
                    model_adequacy: ModelAdequacyStatus::NotFalsified,
                    consistent_ids,
                    trace,
                    qualified_trials: qualified_trials.len(),
                    cost: total_cost,
                    risk: total_risk,
                    validation_count: scenario.validation_probe_ids.len(),
                    validation_predictions: 0,
                    validation_correct: 0,
                    model_inadequacy_detected: false,
                    safe_defer: false,
                    refinement_accepted: false,
                    findings,
                })
            }
        }
        ModelAdequacyDecision::ExpandModelClass { proposal } => {
            if !model_inadequacy_detected {
                findings.retain(|finding| {
                    finding.code != ModelRefinementFindingCode::ExistingModelStillFitsTrace
                });
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::ExpandedAdequateModelClass,
                    None,
                    None,
                    "a supplied model remains consistent; expansion is unsupported by the evidence",
                );
                return report(ModelRefinementReportInput {
                    qualification: ModelRefinementQualification::NotQualified,
                    model_adequacy: ModelAdequacyStatus::NotFalsified,
                    consistent_ids,
                    trace,
                    qualified_trials: qualified_trials.len(),
                    cost: total_cost,
                    risk: total_risk,
                    validation_count: scenario.validation_probe_ids.len(),
                    validation_predictions: 0,
                    validation_correct: 0,
                    model_inadequacy_detected: false,
                    safe_defer: false,
                    refinement_accepted: false,
                    findings,
                });
            }

            let proposal_id = &proposal.hypothesis.id;
            if proposal_id.0.trim().is_empty()
                || proposal.hypothesis.target_class.trim().is_empty()
                || proposal.hypothesis.prior_weight == 0
            {
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::EmptyProposalPrediction,
                    None,
                    None,
                    "proposal requires non-empty ID/target class and positive prior weight",
                );
            }
            if models.contains_key(proposal_id) {
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::RefinementIdAlreadyExists,
                    None,
                    None,
                    "refinement must introduce a distinct hypothesis ID",
                );
            }
            if proposal.hypothesis.target_class != oracle.actual_target_class {
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::ProposalTargetClassMismatch,
                    None,
                    None,
                    "proposal target-equivalence class differs from evaluator-only truth",
                );
            }

            let probe_ids = probes.keys().cloned().collect::<BTreeSet<_>>();
            let prediction_ids = proposal
                .predicted_outcomes
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            if prediction_ids != probe_ids {
                push_finding(
                    &mut findings,
                    ModelRefinementFindingCode::InvalidProposalCoverage,
                    None,
                    None,
                    "proposal must predict every declared probe and no undeclared probe",
                );
            }
            for probe_id in &probe_ids {
                match proposal.predicted_outcomes.get(probe_id) {
                    None => push_finding(
                        &mut findings,
                        ModelRefinementFindingCode::MissingProposalPrediction,
                        None,
                        Some(probe_id.clone()),
                        "proposal omitted a declared probe prediction",
                    ),
                    Some(value) if value.trim().is_empty() => push_finding(
                        &mut findings,
                        ModelRefinementFindingCode::EmptyProposalPrediction,
                        None,
                        Some(probe_id.clone()),
                        "proposal prediction is empty",
                    ),
                    Some(_) => {}
                }
            }
            for probe_id in proposal.predicted_outcomes.keys() {
                if !probes.contains_key(probe_id) {
                    push_finding(
                        &mut findings,
                        ModelRefinementFindingCode::UnknownProposalPrediction,
                        None,
                        Some(probe_id.clone()),
                        "proposal predicts an undeclared probe",
                    );
                }
            }
            for trial in &qualified_trials {
                if proposal
                    .predicted_outcomes
                    .get(&trial.probe_id)
                    .is_none_or(|prediction| prediction != &trial.observed_outcome)
                {
                    push_finding(
                        &mut findings,
                        ModelRefinementFindingCode::ProposalDoesNotExplainDiscovery,
                        None,
                        Some(trial.probe_id.clone()),
                        "proposed model does not explain every qualified discovery observation",
                    );
                }
            }

            let mut validation_predictions = 0;
            let mut validation_correct = 0;
            for probe_id in &scenario.validation_probe_ids {
                if let Some(prediction) = proposal.predicted_outcomes.get(probe_id) {
                    validation_predictions += 1;
                    if oracle
                        .actual_outcomes
                        .get(probe_id)
                        .is_some_and(|actual| actual == prediction)
                    {
                        validation_correct += 1;
                    } else {
                        push_finding(
                            &mut findings,
                            ModelRefinementFindingCode::ValidationPredictionMismatch,
                            None,
                            Some(probe_id.clone()),
                            "proposal failed a sealed validation prediction",
                        );
                    }
                }
            }

            let accepted = findings.is_empty()
                && validation_predictions == scenario.validation_probe_ids.len()
                && validation_correct == scenario.validation_probe_ids.len();
            let final_adequacy = if accepted {
                ModelAdequacyStatus::RefinedAndValidated
            } else {
                ModelAdequacyStatus::Falsified
            };
            report(ModelRefinementReportInput {
                qualification: if accepted {
                    ModelRefinementQualification::Qualified
                } else {
                    ModelRefinementQualification::NotQualified
                },
                model_adequacy: final_adequacy,
                consistent_ids,
                trace,
                qualified_trials: qualified_trials.len(),
                cost: total_cost,
                risk: total_risk,
                validation_count: scenario.validation_probe_ids.len(),
                validation_predictions,
                validation_correct,
                model_inadequacy_detected: true,
                safe_defer: false,
                refinement_accepted: accepted,
                findings,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> (ModelAdequacyScenario, ModelRefinementOracle) {
        let probes = vec![
            RefinementProbe::new("discover-1", 1, 1),
            RefinementProbe::new("discover-2", 1, 1),
            RefinementProbe::new("validate-1", 1, 1),
            RefinementProbe::new("validate-2", 1, 1),
        ];
        let model_a = CandidateMechanismModel::new(
            MechanismHypothesis::new("known-a", "old-class-a", 2),
            [
                (ProbeId::from("discover-1"), "x".into()),
                (ProbeId::from("discover-2"), "y".into()),
                (ProbeId::from("validate-1"), "u".into()),
                (ProbeId::from("validate-2"), "v".into()),
            ]
            .into(),
        );
        let model_b = CandidateMechanismModel::new(
            MechanismHypothesis::new("known-b", "old-class-b", 1),
            [
                (ProbeId::from("discover-1"), "m".into()),
                (ProbeId::from("discover-2"), "n".into()),
                (ProbeId::from("validate-1"), "p".into()),
                (ProbeId::from("validate-2"), "q".into()),
            ]
            .into(),
        );
        let scenario = ModelAdequacyScenario {
            known_models: vec![model_a, model_b],
            probes,
            validation_probe_ids: ["validate-1".into(), "validate-2".into()].into(),
            budget: ExperimentBudget {
                max_cost_units: 4,
                max_risk_units: 4,
                max_trials: 2,
            },
        };
        let oracle = ModelRefinementOracle::new(
            "new-class",
            [
                (ProbeId::from("discover-1"), "z".into()),
                (ProbeId::from("discover-2"), "w".into()),
                (ProbeId::from("validate-1"), "hold-a".into()),
                (ProbeId::from("validate-2"), "hold-b".into()),
            ]
            .into(),
        );
        (scenario, oracle)
    }

    fn trial(probe: &str, outcome: &str) -> ExperimentTrial {
        ExperimentTrial::qualified(probe, outcome)
    }

    fn valid_proposal() -> ModelRefinementProposal {
        ModelRefinementProposal::new(
            MechanismHypothesis::new("new-model", "new-class", 1),
            [
                (ProbeId::from("discover-1"), "z".into()),
                (ProbeId::from("discover-2"), "w".into()),
                (ProbeId::from("validate-1"), "hold-a".into()),
                (ProbeId::from("validate-2"), "hold-b".into()),
            ]
            .into(),
        )
    }

    #[test]
    fn validated_refinement_fits_discovery_and_sealed_probes() {
        let (scenario, oracle) = scenario();
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "z"), trial("discover-2", "w")],
            decision: ModelAdequacyDecision::ExpandModelClass {
                proposal: valid_proposal(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::Qualified,
            "{:?}",
            result.findings
        );
        assert!(result.validates_new_model());
        assert_eq!(
            result.model_adequacy,
            ModelAdequacyStatus::RefinedAndValidated
        );
        assert_eq!(result.validation_accuracy, Some(1.0));
    }

    #[test]
    fn fit_only_refinement_that_fails_a_holdout_is_rejected() {
        let (scenario, oracle) = scenario();
        let mut proposal = valid_proposal();
        proposal
            .predicted_outcomes
            .insert("validate-2".into(), "memorized".into());
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "z"), trial("discover-2", "w")],
            decision: ModelAdequacyDecision::ExpandModelClass { proposal },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::NotQualified
        );
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.code == ModelRefinementFindingCode::ValidationPredictionMismatch)
        );
        assert_eq!(result.validation_accuracy, Some(0.5));
    }

    #[test]
    fn an_existing_model_that_fits_the_complete_trace_should_be_retained() {
        let (scenario, mut oracle) = scenario();
        oracle.actual_target_class = "old-class-a".into();
        oracle
            .actual_outcomes
            .insert("discover-1".into(), "x".into());
        oracle
            .actual_outcomes
            .insert("discover-2".into(), "y".into());
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "x"), trial("discover-2", "y")],
            decision: ModelAdequacyDecision::RetainExistingModel {
                rationale: "one candidate explains both observations".into(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::Qualified,
            "{:?}",
            result.findings
        );
        assert!(result.correctly_retains_model_class());
        assert_eq!(
            result.initial_consistent_model_ids,
            vec![HypothesisId::from("known-a")]
        );
    }

    #[test]
    fn unnecessary_model_expansion_is_rejected() {
        let (scenario, mut oracle) = scenario();
        oracle.actual_target_class = "old-class-a".into();
        oracle
            .actual_outcomes
            .insert("discover-1".into(), "x".into());
        oracle
            .actual_outcomes
            .insert("discover-2".into(), "y".into());
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "x")],
            decision: ModelAdequacyDecision::ExpandModelClass {
                proposal: valid_proposal(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::NotQualified
        );
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.code == ModelRefinementFindingCode::ExpandedAdequateModelClass)
        );
    }

    #[test]
    fn unavailable_delivery_never_falsifies_the_model_class() {
        let (scenario, oracle) = scenario();
        let mut observation = trial("discover-1", "z");
        observation.delivery = ExperimentEvidenceStatus::Unavailable;
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![observation],
            decision: ModelAdequacyDecision::DeferAsUnresolved {
                rationale: "delivery receipt unavailable".into(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::Inconclusive
        );
        assert_eq!(result.model_adequacy, ModelAdequacyStatus::Unresolved);
        assert!(!result.model_inadequacy_detected);
    }

    #[test]
    fn observer_disagreement_with_world_oracle_is_not_called_model_inadequacy() {
        let (scenario, oracle) = scenario();
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "observer-corruption")],
            decision: ModelAdequacyDecision::DeferAsUnresolved {
                rationale: "observation mismatch".into(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::Inconclusive
        );
        assert_eq!(result.model_adequacy, ModelAdequacyStatus::Unresolved);
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.code == ModelRefinementFindingCode::ObservationContradictsOracle)
        );
    }

    #[test]
    fn model_falsification_uses_the_complete_trace_not_individual_observations() {
        let (scenario, mut oracle) = scenario();
        oracle.actual_target_class = "old-class-a".into();
        oracle
            .actual_outcomes
            .insert("discover-1".into(), "x".into());
        oracle
            .actual_outcomes
            .insert("discover-2".into(), "n".into());
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "x"), trial("discover-2", "n")],
            decision: ModelAdequacyDecision::DeferAsUnresolved {
                rationale: "no complete model fits".into(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(result.model_adequacy, ModelAdequacyStatus::Falsified);
        assert!(result.model_inadequacy_detected);
        assert!(result.safe_defer);
    }

    #[test]
    fn cost_and_risk_budgets_block_model_adequacy_claims() {
        let (mut scenario, oracle) = scenario();
        scenario.budget.max_risk_units = 0;
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("discover-1", "z")],
            decision: ModelAdequacyDecision::ExpandModelClass {
                proposal: valid_proposal(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::NotQualified
        );
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.code == ModelRefinementFindingCode::RiskBudgetExceeded)
        );
    }

    #[test]
    fn malformed_incomplete_known_model_predictions_invalidate_the_scenario() {
        let (mut scenario, oracle) = scenario();
        scenario.known_models[0]
            .predicted_outcomes
            .remove(&ProbeId::from("validate-1"));
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![],
            decision: ModelAdequacyDecision::DeferAsUnresolved {
                rationale: "invalid scenario".into(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::Inconclusive
        );
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.code == ModelRefinementFindingCode::MissingModelPrediction)
        );
    }

    #[test]
    fn sealed_validation_probes_cannot_be_used_during_discovery() {
        let (scenario, oracle) = scenario();
        let trace = ModelAdequacyTrace {
            discovery_trials: vec![trial("validate-1", "hold-a")],
            decision: ModelAdequacyDecision::DeferAsUnresolved {
                rationale: "invalid use of holdout".into(),
            },
        };
        let result = evaluate_model_refinement(&scenario, &oracle, &trace);
        assert_eq!(
            result.qualification,
            ModelRefinementQualification::NotQualified
        );
        assert!(
            result
                .findings
                .iter()
                .any(|f| f.code == ModelRefinementFindingCode::UnknownObservedProbe)
        );
    }
}
