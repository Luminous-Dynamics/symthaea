//! Sequential deterministic experiment design for AOBench.
//!
//! Evaluates adaptive probe choice, hypothesis updates, intervention evidence,
//! resource budgets and stop/abstain decisions over a frozen finite model.
//! This pilot is deterministic: it does not implement stochastic power,
//! confidence intervals, multiplicity or unrestricted causal inference.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HypothesisId(pub String);
impl From<&str> for HypothesisId {
    fn from(value: &str) -> Self { Self(value.to_owned()) }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProbeId(pub String);
impl From<&str> for ProbeId {
    fn from(value: &str) -> Self { Self(value.to_owned()) }
}

/// Candidate mechanism and its prior weight in the frozen finite model class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismHypothesis {
    pub id: HypothesisId,
    pub target_class: String,
    pub prior_weight: u64,
}
impl MechanismHypothesis {
    pub fn new(id: impl Into<HypothesisId>, target_class: impl Into<String>, prior_weight: u64) -> Self {
        Self { id: id.into(), target_class: target_class.into(), prior_weight }
    }
}

/// Deterministic response table. Every hypothesis must have one non-empty outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentProbe {
    pub id: ProbeId,
    pub cost_units: u64,
    pub risk_units: u64,
    pub outcomes: BTreeMap<HypothesisId, String>,
}
impl ExperimentProbe {
    pub fn new(id: impl Into<ProbeId>, cost_units: u64, risk_units: u64,
        outcomes: BTreeMap<HypothesisId, String>) -> Self {
        Self { id: id.into(), cost_units, risk_units, outcomes }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentBudget {
    pub max_cost_units: u64,
    pub max_risk_units: u64,
    pub max_trials: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialExperimentScenario {
    pub hypotheses: Vec<MechanismHypothesis>,
    pub probes: Vec<ExperimentProbe>,
    pub budget: ExperimentBudget,
}

/// Evaluator-only ground truth. Never expose this to the experimenter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentOracle {
    pub actual_hypothesis: HypothesisId,
}
impl ExperimentOracle {
    pub fn new(actual_hypothesis: impl Into<HypothesisId>) -> Self {
        Self { actual_hypothesis: actual_hypothesis.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentEvidenceStatus { Verified, Rejected, Unavailable }

/// Delivery, isolation and observation are independent evidence gates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentTrial {
    pub probe_id: ProbeId,
    pub observed_outcome: String,
    pub delivery: ExperimentEvidenceStatus,
    pub isolation: ExperimentEvidenceStatus,
    pub observation: ExperimentEvidenceStatus,
}
impl ExperimentTrial {
    pub fn qualified(probe_id: impl Into<ProbeId>, observed_outcome: impl Into<String>) -> Self {
        Self {
            probe_id: probe_id.into(),
            observed_outcome: observed_outcome.into(),
            delivery: ExperimentEvidenceStatus::Verified,
            isolation: ExperimentEvidenceStatus::Verified,
            observation: ExperimentEvidenceStatus::Verified,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExperimentTerminalDecision {
    Stop { target_class: String },
    Abstain { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialExperimentTrace {
    pub trials: Vec<ExperimentTrial>,
    pub terminal: ExperimentTerminalDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentQualification { Qualified, NotQualified, Inconclusive }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentFindingCode {
    EmptyHypothesisSet,
    DuplicateHypothesisId,
    EmptyHypothesisId,
    EmptyTargetClass,
    ZeroPriorWeight,
    UnknownOracleHypothesis,
    EmptyProbeId,
    DuplicateProbeId,
    MissingProbeResponse,
    UnknownProbeResponseHypothesis,
    EmptyProbeOutcome,
    UnknownProbe,
    EmptyObservedOutcome,
    CostBudgetExceeded,
    RiskBudgetExceeded,
    TrialBudgetExceeded,
    DeliveryRejected,
    IsolationRejected,
    ObservationRejected,
    EvidenceUnavailable,
    ObservationContradictsFrozenModel,
    NoConsistentHypothesis,
    ProbeSelectedAfterIdentification,
    UninformativeProbe,
    PrematureStop,
    WrongTargetClass,
    AbstainedAfterIdentification,
    AbstainedWhileInformativeProbeAvailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentFinding {
    pub code: ExperimentFindingCode,
    pub trial_index: Option<usize>,
    pub probe_id: Option<ProbeId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialInformationAssessment {
    pub trial_index: usize,
    pub probe_id: ProbeId,
    pub target_classes_before: usize,
    pub information_gain_bits: f64,
    pub best_available_information_gain_bits: f64,
    pub information_gain_per_cost_unit: f64,
    pub best_available_information_gain_per_cost_unit: f64,
    pub evidence_qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialExperimentReport {
    pub qualification: ExperimentQualification,
    pub remaining_hypotheses: Vec<HypothesisId>,
    pub remaining_target_classes: Vec<String>,
    pub total_cost_units: u64,
    pub total_risk_units: u64,
    pub trials_used: usize,
    pub qualified_trial_count: usize,
    pub unqualified_trial_count: usize,
    pub informative_trial_count: usize,
    pub first_identified_after_trials: Option<usize>,
    pub cumulative_information_gain_bits: f64,
    pub cumulative_oracle_regret_bits_per_cost: f64,
    pub trial_assessments: Vec<TrialInformationAssessment>,
    pub findings: Vec<ExperimentFinding>,
}
impl SequentialExperimentReport {
    pub fn correctly_identified_target_class(&self) -> bool {
        self.qualification == ExperimentQualification::Qualified
            && self.remaining_target_classes.len() == 1
            && self.findings.is_empty()
    }
    pub fn correct_abstention(&self) -> bool {
        self.qualification == ExperimentQualification::Qualified
            && self.remaining_target_classes.len() > 1
            && self.findings.is_empty()
    }
}

fn push_finding(findings: &mut Vec<ExperimentFinding>, code: ExperimentFindingCode,
    trial_index: Option<usize>, probe_id: Option<ProbeId>, detail: impl Into<String>) {
    findings.push(ExperimentFinding { code, trial_index, probe_id, detail: detail.into() });
}

fn qualification(findings: &[ExperimentFinding], invalid_scenario: bool) -> ExperimentQualification {
    if invalid_scenario {
        return ExperimentQualification::Inconclusive;
    }
    let only_inconclusive_findings = findings.iter().all(|finding| matches!(
        finding.code,
        ExperimentFindingCode::EvidenceUnavailable
            | ExperimentFindingCode::ObservationContradictsFrozenModel
            | ExperimentFindingCode::NoConsistentHypothesis
    ));
    if !findings.is_empty() && only_inconclusive_findings {
        ExperimentQualification::Inconclusive
    } else if !findings.is_empty() {
        ExperimentQualification::NotQualified
    } else {
        ExperimentQualification::Qualified
    }
}

fn target_classes(active: &BTreeSet<HypothesisId>,
    hypotheses: &BTreeMap<HypothesisId, &MechanismHypothesis>) -> BTreeSet<String> {
    active.iter().filter_map(|id| hypotheses.get(id))
        .map(|hypothesis| hypothesis.target_class.clone()).collect()
}

fn entropy(weights: &BTreeMap<String, f64>) -> f64 {
    let total: f64 = weights.values().sum();
    if total <= 0.0 { return 0.0; }
    weights.values().filter(|weight| **weight > 0.0).map(|weight| {
        let p = *weight / total;
        -p * p.log2()
    }).sum()
}

/// Expected reduction in target-class entropy under one frozen probe.
fn information_gain_bits(active: &BTreeSet<HypothesisId>,
    hypotheses: &BTreeMap<HypothesisId, &MechanismHypothesis>,
    probe: &ExperimentProbe) -> f64 {
    let mut prior_by_class: BTreeMap<String, f64> = BTreeMap::new();
    let mut by_outcome_and_class: BTreeMap<(String, String), f64> = BTreeMap::new();
    let mut outcome_totals: BTreeMap<String, f64> = BTreeMap::new();

    for id in active {
        let Some(hypothesis) = hypotheses.get(id) else { continue };
        let weight = hypothesis.prior_weight as f64;
        *prior_by_class.entry(hypothesis.target_class.clone()).or_default() += weight;
        if let Some(outcome) = probe.outcomes.get(id) {
            *by_outcome_and_class.entry((outcome.clone(), hypothesis.target_class.clone())).or_default() += weight;
            *outcome_totals.entry(outcome.clone()).or_default() += weight;
        }
    }
    let total: f64 = prior_by_class.values().sum();
    if total <= 0.0 { return 0.0; }
    let prior_entropy = entropy(&prior_by_class);
    let mut expected_posterior_entropy = 0.0;
    for (outcome, outcome_weight) in &outcome_totals {
        let mut conditional_classes: BTreeMap<String, f64> = BTreeMap::new();
        for ((observed, class), weight) in &by_outcome_and_class {
            if observed == outcome { conditional_classes.insert(class.clone(), *weight); }
        }
        expected_posterior_entropy += (*outcome_weight / total) * entropy(&conditional_classes);
    }
    (prior_entropy - expected_posterior_entropy).max(0.0)
}

fn eligible_probe_utilities(active: &BTreeSet<HypothesisId>,
    hypotheses: &BTreeMap<HypothesisId, &MechanismHypothesis>,
    probes: &BTreeMap<ProbeId, &ExperimentProbe>, remaining_cost: u64,
    remaining_risk: u64, remaining_trials: usize) -> Vec<(ProbeId, f64, f64)> {
    if remaining_trials == 0 { return Vec::new(); }
    probes.values()
        .filter(|probe| probe.cost_units <= remaining_cost && probe.risk_units <= remaining_risk)
        .map(|probe| {
            let gain = information_gain_bits(active, hypotheses, probe);
            let efficiency = gain / probe.cost_units.max(1) as f64;
            (probe.id.clone(), gain, efficiency)
        })
        .collect()
}

fn invalid_scenario_report(findings: Vec<ExperimentFinding>,
    active: BTreeSet<HypothesisId>,
    hypotheses: &BTreeMap<HypothesisId, &MechanismHypothesis>,
    trace: &SequentialExperimentTrace) -> SequentialExperimentReport {
    SequentialExperimentReport {
        qualification: ExperimentQualification::Inconclusive,
        remaining_target_classes: target_classes(&active, hypotheses).into_iter().collect(),
        remaining_hypotheses: active.into_iter().collect(),
        total_cost_units: 0,
        total_risk_units: 0,
        trials_used: trace.trials.len(),
        qualified_trial_count: 0,
        unqualified_trial_count: 0,
        informative_trial_count: 0,
        first_identified_after_trials: None,
        cumulative_information_gain_bits: 0.0,
        cumulative_oracle_regret_bits_per_cost: 0.0,
        trial_assessments: Vec::new(),
        findings,
    }
}

/// Evaluate a sequential probe trace against a finite response model.
///
/// The agent receives the scenario and emits the trace. The evaluator-only
/// oracle remains hidden. Hypotheses are removed only after delivery, isolation
/// and observation evidence are all verified. Stop is accepted only when one
/// target-equivalence class remains; abstain is accepted when no informative
/// probe remains within the frozen cost, risk and trial budgets.
pub fn evaluate_sequential_experiment(scenario: &SequentialExperimentScenario,
    oracle: &ExperimentOracle, trace: &SequentialExperimentTrace) -> SequentialExperimentReport {
    let mut findings = Vec::new();
    if scenario.hypotheses.is_empty() {
        push_finding(&mut findings, ExperimentFindingCode::EmptyHypothesisSet, None, None,
            "the frozen mechanism hypothesis set is empty");
    }

    let mut hypotheses: BTreeMap<HypothesisId, &MechanismHypothesis> = BTreeMap::new();
    for hypothesis in &scenario.hypotheses {
        if hypothesis.id.0.trim().is_empty() {
            push_finding(&mut findings, ExperimentFindingCode::EmptyHypothesisId, None, None,
                "hypothesis ID is empty");
        }
        if hypothesis.target_class.trim().is_empty() {
            push_finding(&mut findings, ExperimentFindingCode::EmptyTargetClass, None, None,
                "hypothesis target-equivalence class is empty");
        }
        if hypothesis.prior_weight == 0 {
            push_finding(&mut findings, ExperimentFindingCode::ZeroPriorWeight, None, None,
                "hypothesis prior weight must be positive");
        }
        if hypotheses.insert(hypothesis.id.clone(), hypothesis).is_some() {
            push_finding(&mut findings, ExperimentFindingCode::DuplicateHypothesisId, None, None,
                "hypothesis ID is duplicated");
        }
    }
    if !hypotheses.contains_key(&oracle.actual_hypothesis) {
        push_finding(&mut findings, ExperimentFindingCode::UnknownOracleHypothesis, None, None,
            "evaluator-only actual hypothesis is absent from the frozen model class");
    }

    let mut probes: BTreeMap<ProbeId, &ExperimentProbe> = BTreeMap::new();
    for probe in &scenario.probes {
        if probe.id.0.trim().is_empty() {
            push_finding(&mut findings, ExperimentFindingCode::EmptyProbeId, None, Some(probe.id.clone()),
                "probe ID is empty");
        }
        if probes.insert(probe.id.clone(), probe).is_some() {
            push_finding(&mut findings, ExperimentFindingCode::DuplicateProbeId, None, Some(probe.id.clone()),
                "probe ID is duplicated");
        }
        for id in hypotheses.keys() {
            match probe.outcomes.get(id) {
                None => push_finding(&mut findings, ExperimentFindingCode::MissingProbeResponse, None,
                    Some(probe.id.clone()), format!("probe has no response for hypothesis {}", id.0)),
                Some(outcome) if outcome.trim().is_empty() => push_finding(&mut findings,
                    ExperimentFindingCode::EmptyProbeOutcome, None, Some(probe.id.clone()),
                    format!("probe has an empty response for hypothesis {}", id.0)),
                Some(_) => {}
            }
        }
        for id in probe.outcomes.keys() {
            if !hypotheses.contains_key(id) {
                push_finding(&mut findings, ExperimentFindingCode::UnknownProbeResponseHypothesis, None,
                    Some(probe.id.clone()), format!("probe response references unknown hypothesis {}", id.0));
            }
        }
    }

    let scenario_invalid = !findings.is_empty();
    let mut active: BTreeSet<HypothesisId> = hypotheses.keys().cloned().collect();
    if scenario_invalid {
        return invalid_scenario_report(findings, active, &hypotheses, trace);
    }

    let mut total_cost = 0_u64;
    let mut total_risk = 0_u64;
    let mut qualified_trials = 0_usize;
    let mut unqualified_trials = 0_usize;
    let mut informative_trials = 0_usize;
    let mut first_identified_after_trials =
        if target_classes(&active, &hypotheses).len() == 1 { Some(0) } else { None };
    let mut cumulative_information_gain = 0.0_f64;
    let mut cumulative_oracle_regret = 0.0_f64;
    let mut assessments = Vec::new();

    for (index, trial) in trace.trials.iter().enumerate() {
        let trial_index = index + 1;
        if trial_index > scenario.budget.max_trials {
            push_finding(&mut findings, ExperimentFindingCode::TrialBudgetExceeded,
                Some(trial_index), Some(trial.probe_id.clone()), "trace exceeds the frozen trial budget");
            break;
        }
        let Some(probe) = probes.get(&trial.probe_id) else {
            push_finding(&mut findings, ExperimentFindingCode::UnknownProbe,
                Some(trial_index), Some(trial.probe_id.clone()), "trial references an undeclared probe");
            continue;
        };
        if first_identified_after_trials.is_some() {
            push_finding(&mut findings, ExperimentFindingCode::ProbeSelectedAfterIdentification,
                Some(trial_index), Some(trial.probe_id.clone()),
                "a target-equivalence class was already identified before this probe");
        }

        let cost_before = total_cost;
        let risk_before = total_risk;
        let next_cost = total_cost.checked_add(probe.cost_units);
        let next_risk = total_risk.checked_add(probe.risk_units);
        if next_cost.is_none_or(|value| value > scenario.budget.max_cost_units) {
            push_finding(&mut findings, ExperimentFindingCode::CostBudgetExceeded,
                Some(trial_index), Some(trial.probe_id.clone()),
                "probe would exceed the frozen cumulative cost budget");
        }
        if next_risk.is_none_or(|value| value > scenario.budget.max_risk_units) {
            push_finding(&mut findings, ExperimentFindingCode::RiskBudgetExceeded,
                Some(trial_index), Some(trial.probe_id.clone()),
                "probe would exceed the frozen cumulative risk budget");
        }
        total_cost = next_cost.unwrap_or(u64::MAX);
        total_risk = next_risk.unwrap_or(u64::MAX);

        let utilities = eligible_probe_utilities(
            &active, &hypotheses, &probes,
            scenario.budget.max_cost_units.saturating_sub(cost_before),
            scenario.budget.max_risk_units.saturating_sub(risk_before),
            scenario.budget.max_trials.saturating_sub(index),
        );
        let best_gain = utilities.iter().map(|(_, gain, _)| *gain).fold(0.0_f64, f64::max);
        let best_efficiency = utilities.iter().map(|(_, _, value)| *value).fold(0.0_f64, f64::max);
        let chosen_gain = information_gain_bits(&active, &hypotheses, probe);
        let chosen_efficiency = chosen_gain / probe.cost_units.max(1) as f64;
        cumulative_information_gain += chosen_gain;
        cumulative_oracle_regret += (best_efficiency - chosen_efficiency).max(0.0);
        if chosen_gain > f64::EPSILON {
            informative_trials += 1;
        }

        let mut evidence_qualified = true;
        for (label, status) in [
            ("delivery", trial.delivery),
            ("isolation", trial.isolation),
            ("observation", trial.observation),
        ] {
            match status {
                ExperimentEvidenceStatus::Verified => {}
                ExperimentEvidenceStatus::Rejected => {
                    evidence_qualified = false;
                    let code = match label {
                        "delivery" => ExperimentFindingCode::DeliveryRejected,
                        "isolation" => ExperimentFindingCode::IsolationRejected,
                        _ => ExperimentFindingCode::ObservationRejected,
                    };
                    push_finding(&mut findings, code, Some(trial_index), Some(trial.probe_id.clone()),
                        format!("{label} evidence was rejected"));
                }
                ExperimentEvidenceStatus::Unavailable => {
                    evidence_qualified = false;
                    push_finding(&mut findings, ExperimentFindingCode::EvidenceUnavailable,
                        Some(trial_index), Some(trial.probe_id.clone()),
                        format!("{label} evidence is unavailable; outcome cannot update hypotheses"));
                }
            }
        }
        if trial.observed_outcome.trim().is_empty() {
            evidence_qualified = false;
            push_finding(&mut findings, ExperimentFindingCode::EmptyObservedOutcome,
                Some(trial_index), Some(trial.probe_id.clone()), "observed outcome is empty");
        }

        assessments.push(TrialInformationAssessment {
            trial_index,
            probe_id: trial.probe_id.clone(),
            target_classes_before: target_classes(&active, &hypotheses).len(),
            information_gain_bits: chosen_gain,
            best_available_information_gain_bits: best_gain,
            information_gain_per_cost_unit: chosen_efficiency,
            best_available_information_gain_per_cost_unit: best_efficiency,
            evidence_qualified,
        });
        if !evidence_qualified {
            unqualified_trials += 1;
            continue;
        }
        qualified_trials += 1;

        if probe.outcomes.get(&oracle.actual_hypothesis)
            .is_none_or(|expected| expected != &trial.observed_outcome) {
            push_finding(&mut findings, ExperimentFindingCode::ObservationContradictsFrozenModel,
                Some(trial_index), Some(trial.probe_id.clone()),
                "qualified observation differs from the evaluator-only deterministic response");
            continue;
        }

        if chosen_gain <= f64::EPSILON && target_classes(&active, &hypotheses).len() > 1 {
            push_finding(&mut findings, ExperimentFindingCode::UninformativeProbe,
                Some(trial_index), Some(trial.probe_id.clone()),
                "selected probe does not distinguish the remaining target classes under the frozen model");
        }

        active.retain(|id| probe.outcomes.get(id)
            .is_some_and(|outcome| outcome == &trial.observed_outcome));
        if active.is_empty() || !active.contains(&oracle.actual_hypothesis) {
            push_finding(&mut findings, ExperimentFindingCode::NoConsistentHypothesis,
                Some(trial_index), Some(trial.probe_id.clone()),
                "qualified observation leaves no consistent hypothesis including evaluator-only truth");
            continue;
        }
        if first_identified_after_trials.is_none()
            && target_classes(&active, &hypotheses).len() == 1 {
            first_identified_after_trials = Some(trial_index);
        }
    }

    let remaining_classes = target_classes(&active, &hypotheses);
    let actual_class = hypotheses.get(&oracle.actual_hypothesis)
        .map(|hypothesis| hypothesis.target_class.as_str());

    match &trace.terminal {
        ExperimentTerminalDecision::Stop { target_class } => {
            if remaining_classes.len() != 1 {
                push_finding(&mut findings, ExperimentFindingCode::PrematureStop, None, None,
                    "more than one target-equivalence class remains");
            } else if remaining_classes.iter().next().is_some_and(|class| class != target_class)
                || actual_class.is_none_or(|class| class != target_class) {
                push_finding(&mut findings, ExperimentFindingCode::WrongTargetClass, None, None,
                    "declared target class differs from the uniquely identified or evaluator-only class");
            }
        }
        ExperimentTerminalDecision::Abstain { reason: _ } => {
            if remaining_classes.len() == 1 {
                push_finding(&mut findings, ExperimentFindingCode::AbstainedAfterIdentification, None, None,
                    "agent abstained after a single target-equivalence class was identified");
            } else if remaining_classes.len() > 1 {
                let remaining_cost = scenario.budget.max_cost_units.saturating_sub(total_cost);
                let remaining_risk = scenario.budget.max_risk_units.saturating_sub(total_risk);
                let remaining_trials = scenario.budget.max_trials.saturating_sub(trace.trials.len());
                let model_class_discrepancy = findings.iter().any(|finding| matches!(
                    finding.code,
                    ExperimentFindingCode::ObservationContradictsFrozenModel
                        | ExperimentFindingCode::NoConsistentHypothesis
                ));
                let informative_available = !model_class_discrepancy
                    && eligible_probe_utilities(
                        &active, &hypotheses, &probes, remaining_cost, remaining_risk, remaining_trials,
                    ).iter().any(|(_, gain, _)| *gain > f64::EPSILON);
                if informative_available {
                    push_finding(&mut findings, ExperimentFindingCode::AbstainedWhileInformativeProbeAvailable,
                        None, None, "agent abstained while an informative probe remained within all frozen budgets");
                }
            }
        }
    }

    SequentialExperimentReport {
        qualification: qualification(&findings, false),
        remaining_hypotheses: active.into_iter().collect(),
        remaining_target_classes: remaining_classes.into_iter().collect(),
        total_cost_units: total_cost,
        total_risk_units: total_risk,
        trials_used: trace.trials.len(),
        qualified_trial_count: qualified_trials,
        unqualified_trial_count: unqualified_trials,
        informative_trial_count: informative_trials,
        first_identified_after_trials,
        cumulative_information_gain_bits: cumulative_information_gain,
        cumulative_oracle_regret_bits_per_cost: cumulative_oracle_regret,
        trial_assessments: assessments,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> SequentialExperimentScenario {
        SequentialExperimentScenario {
            hypotheses: vec![
                MechanismHypothesis::new("model-fault", "model-corruption", 3),
                MechanismHypothesis::new("memory-fault", "memory-staleness", 2),
                MechanismHypothesis::new("dependency-fault", "external-dependency", 1),
            ],
            probes: vec![
                ExperimentProbe::new("inspect-model", 2, 1, [
                    (HypothesisId::from("model-fault"), "corrupt".into()),
                    (HypothesisId::from("memory-fault"), "healthy".into()),
                    (HypothesisId::from("dependency-fault"), "healthy".into()),
                ].into()),
                ExperimentProbe::new("inspect-memory", 1, 1, [
                    (HypothesisId::from("model-fault"), "healthy".into()),
                    (HypothesisId::from("memory-fault"), "stale".into()),
                    (HypothesisId::from("dependency-fault"), "healthy".into()),
                ].into()),
                ExperimentProbe::new("inspect-boundary", 3, 2, [
                    (HypothesisId::from("model-fault"), "open".into()),
                    (HypothesisId::from("memory-fault"), "open".into()),
                    (HypothesisId::from("dependency-fault"), "blocked".into()),
                ].into()),
            ],
            budget: ExperimentBudget { max_cost_units: 5, max_risk_units: 3, max_trials: 3 },
        }
    }

    #[test]
    fn sequential_probes_identify_hidden_mechanism_and_stop() {
        let scenario = scenario();
        let oracle = ExperimentOracle::new("model-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![
                ExperimentTrial::qualified("inspect-memory", "healthy"),
                ExperimentTrial::qualified("inspect-model", "corrupt"),
            ],
            terminal: ExperimentTerminalDecision::Stop { target_class: "model-corruption".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::Qualified);
        assert!(report.correctly_identified_target_class());
        assert_eq!(report.remaining_hypotheses, vec![HypothesisId::from("model-fault")]);
        assert_eq!(report.first_identified_after_trials, Some(2));
        assert_eq!(report.qualified_trial_count, 2);
        assert!(report.cumulative_information_gain_bits > 0.0);
    }

    #[test]
    fn premature_stop_is_rejected() {
        let scenario = scenario();
        let oracle = ExperimentOracle::new("memory-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![ExperimentTrial::qualified("inspect-model", "healthy")],
            terminal: ExperimentTerminalDecision::Stop { target_class: "memory-staleness".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == ExperimentFindingCode::PrematureStop));
    }

    #[test]
    fn unavailable_evidence_never_updates_hypotheses() {
        let mut scenario = scenario();
        scenario.budget.max_trials = 1;
        let oracle = ExperimentOracle::new("memory-fault");
        let mut trial = ExperimentTrial::qualified("inspect-model", "healthy");
        trial.delivery = ExperimentEvidenceStatus::Unavailable;
        let trace = SequentialExperimentTrace {
            trials: vec![trial],
            terminal: ExperimentTerminalDecision::Abstain { reason: "delivery receipt unavailable".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::Inconclusive);
        assert_eq!(report.remaining_hypotheses.len(), 3);
        assert_eq!(report.unqualified_trial_count, 1);
    }

    #[test]
    fn qualified_observation_outside_frozen_model_is_inconclusive() {
        let scenario = scenario();
        let oracle = ExperimentOracle::new("memory-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![ExperimentTrial::qualified("inspect-model", "unexpected")],
            terminal: ExperimentTerminalDecision::Abstain { reason: "model mismatch".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::Inconclusive);
        assert!(report.findings.iter().any(|f| f.code == ExperimentFindingCode::ObservationContradictsFrozenModel));
    }

    #[test]
    fn cost_and_risk_budgets_are_non_compensatory() {
        let mut scenario = scenario();
        scenario.budget.max_cost_units = 1;
        let oracle = ExperimentOracle::new("memory-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![ExperimentTrial::qualified("inspect-model", "healthy")],
            terminal: ExperimentTerminalDecision::Abstain { reason: "budget exhausted".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == ExperimentFindingCode::CostBudgetExceeded));
    }

    #[test]
    fn models_in_the_same_target_class_do_not_need_to_be_distinguished() {
        let scenario = SequentialExperimentScenario {
            hypotheses: vec![
                MechanismHypothesis::new("implementation-a", "same-capability-class", 1),
                MechanismHypothesis::new("implementation-b", "same-capability-class", 1),
            ],
            probes: vec![],
            budget: ExperimentBudget { max_cost_units: 0, max_risk_units: 0, max_trials: 0 },
        };
        let oracle = ExperimentOracle::new("implementation-a");
        let trace = SequentialExperimentTrace {
            trials: vec![],
            terminal: ExperimentTerminalDecision::Stop {
                target_class: "same-capability-class".into(),
            },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::Qualified);
        assert!(report.correctly_identified_target_class());
        assert_eq!(report.remaining_hypotheses.len(), 2);
        assert_eq!(report.first_identified_after_trials, Some(0));
    }

    #[test]
    fn correct_abstention_when_no_informative_probe_fits_budget_is_qualified() {
        let mut scenario = scenario();
        scenario.budget.max_cost_units = 0;
        scenario.budget.max_risk_units = 0;
        let oracle = ExperimentOracle::new("memory-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![],
            terminal: ExperimentTerminalDecision::Abstain { reason: "no safe affordable probe".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::Qualified);
        assert!(report.correct_abstention());
    }

    #[test]
    fn uninformative_probe_is_reported() {
        let mut scenario = scenario();
        scenario.probes.push(ExperimentProbe::new("no-op", 0, 0, [
            (HypothesisId::from("model-fault"), "same".into()),
            (HypothesisId::from("memory-fault"), "same".into()),
            (HypothesisId::from("dependency-fault"), "same".into()),
        ].into()));
        let oracle = ExperimentOracle::new("memory-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![ExperimentTrial::qualified("no-op", "same")],
            terminal: ExperimentTerminalDecision::Abstain { reason: "no further signal".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == ExperimentFindingCode::UninformativeProbe));
    }

    #[test]
    fn oracle_regret_is_positive_for_a_less_efficient_probe_choice() {
        let scenario = scenario();
        let oracle = ExperimentOracle::new("model-fault");
        let trace = SequentialExperimentTrace {
            trials: vec![ExperimentTrial::qualified("inspect-model", "corrupt")],
            terminal: ExperimentTerminalDecision::Stop { target_class: "model-corruption".into() },
        };
        let report = evaluate_sequential_experiment(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ExperimentQualification::Qualified);
        assert!(report.cumulative_oracle_regret_bits_per_cost > 0.0);
    }
}
