//! Site-disjoint empirical observation-likelihood calibration for Subterra.
//!
//! Estimates categorical P(outcome | hypothesis, action) tables from
//! independently labelled training sites, then scores calibration/evaluation
//! sites using proper scoring rules.
//!
//! This package does not infer priors, authenticate evidence, fit a hazard
//! model, or guarantee performance under distribution shift.

use std::collections::{BTreeMap, BTreeSet};

use subterra_voi_reference::{
    Hypothesis as PlannerHypothesis, MeasurementAction as PlannerMeasurementAction,
    MissionDecision as PlannerMissionDecision, ObservationOutcome as PlannerObservationOutcome,
    PlanningProblem, ResourceCost,
};

pub const UNKNOWN_HYPOTHESIS_ID: &str = "unknown";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataSplit {
    Training,
    Calibration,
    Evaluation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionContract {
    pub action_id: String,
    pub modality: String,
    /// Ordered, exhaustive categorical outcomes for this action.
    pub outcome_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationCase {
    pub case_id: String,
    pub site_id: String,
    /// Unique physical acquisition lineage. Derived/reprocessed products
    /// from one acquisition must not be entered as multiple independent cases.
    pub acquisition_lineage_id: String,
    pub action_id: String,
    /// Independently adjudicated reference class, including explicit unknown.
    pub reference_hypothesis_id: String,
    pub observed_outcome_id: String,
    pub split: DataSplit,
    /// Opaque adjudication provenance; this package does not authenticate it.
    pub reference_label_evidence_ref: String,
    /// Opaque observation provenance; this package does not authenticate it.
    pub observation_evidence_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationDataset {
    pub dataset_id: String,
    pub hypothesis_ids: Vec<String>,
    pub actions: Vec<ActionContract>,
    pub cases: Vec<CalibrationCase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationError {
    EmptyDatasetId,
    EmptyHypothesisSet,
    EmptyHypothesisId,
    MissingUnknownHypothesis,
    DuplicateHypothesisId(String),
    EmptyActionSet,
    EmptyActionId,
    DuplicateActionId(String),
    ActionHasNoOutcomes(String),
    DuplicateOutcomeId { action_id: String, outcome_id: String },
    EmptyCaseSet,
    EmptyField { case_id: String, field: &'static str },
    DuplicateCaseId(String),
    DuplicateAcquisitionLineage(String),
    DuplicateSiteAction { site_id: String, action_id: String },
    SiteAppearsInMultipleSplits(String),
    UnknownAction { case_id: String, action_id: String },
    UnknownHypothesis { case_id: String, hypothesis_id: String },
    UnknownOutcome { case_id: String, action_id: String, outcome_id: String },
    NoTrainingCases,
    InvalidLikelihoodTable,
    PlannerHypothesisOrderMismatch,
    PlannerDecisionDimensionMismatch,
    PlannerActionPolicyMismatch,
    DuplicatePlannerActionPolicy(String),
    InsufficientLikelihoodSupport {
        action_id: String,
        hypothesis_id: String,
        training_sample_count: usize,
    },
    InvalidDirichletAlpha,
    InvalidMinimumTrainingSites,
    CannotScoreTrainingSplit,
    NoCasesForSplit(DataSplit),
    NoScorableCases(DataSplit),
    NumericalOverflow(&'static str),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CalibrationError {}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn duplicate_id<'a>(values: impl IntoIterator<Item = &'a str>) -> Option<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .find(|value| !seen.insert(*value))
        .map(str::to_string)
}

impl CalibrationDataset {
    /// Validate split isolation, case identity, action/hypothesis/outcome
    /// membership and opaque evidence-reference presence.
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if !nonempty(&self.dataset_id) {
            return Err(CalibrationError::EmptyDatasetId);
        }
        if self.hypothesis_ids.is_empty() {
            return Err(CalibrationError::EmptyHypothesisSet);
        }
        if self.hypothesis_ids.iter().any(|id| !nonempty(id)) {
            return Err(CalibrationError::EmptyHypothesisId);
        }
        if !self.hypothesis_ids.iter().any(|id| id == UNKNOWN_HYPOTHESIS_ID) {
            return Err(CalibrationError::MissingUnknownHypothesis);
        }
        if let Some(id) = duplicate_id(self.hypothesis_ids.iter().map(String::as_str)) {
            return Err(CalibrationError::DuplicateHypothesisId(id));
        }
        if self.actions.is_empty() {
            return Err(CalibrationError::EmptyActionSet);
        }

        let mut action_ids = BTreeSet::new();
        for action in &self.actions {
            if !nonempty(&action.action_id) {
                return Err(CalibrationError::EmptyActionId);
            }
            if !action_ids.insert(action.action_id.as_str()) {
                return Err(CalibrationError::DuplicateActionId(action.action_id.clone()));
            }
            if !nonempty(&action.modality) {
                return Err(CalibrationError::EmptyField {
                    case_id: action.action_id.clone(),
                    field: "modality",
                });
            }
            if action.outcome_ids.is_empty() {
                return Err(CalibrationError::ActionHasNoOutcomes(action.action_id.clone()));
            }
            if let Some(outcome_id) =
                duplicate_id(action.outcome_ids.iter().map(String::as_str))
            {
                return Err(CalibrationError::DuplicateOutcomeId {
                    action_id: action.action_id.clone(),
                    outcome_id,
                });
            }
            if action.outcome_ids.iter().any(|id| !nonempty(id)) {
                return Err(CalibrationError::EmptyField {
                    case_id: action.action_id.clone(),
                    field: "outcome_id",
                });
            }
        }

        if self.cases.is_empty() {
            return Err(CalibrationError::EmptyCaseSet);
        }

        let actions = self
            .actions
            .iter()
            .map(|action| (action.action_id.as_str(), action.outcome_ids.as_slice()))
            .collect::<BTreeMap<_, _>>();
        let hypothesis_ids = self
            .hypothesis_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();

        let mut case_ids = BTreeSet::new();
        let mut acquisition_ids = BTreeSet::new();
        let mut site_splits = BTreeMap::<&str, DataSplit>::new();
        let mut site_actions = BTreeSet::<(&str, &str)>::new();

        for case in &self.cases {
            for (value, field) in [
                (case.case_id.as_str(), "case_id"),
                (case.site_id.as_str(), "site_id"),
                (case.acquisition_lineage_id.as_str(), "acquisition_lineage_id"),
                (case.action_id.as_str(), "action_id"),
                (case.reference_hypothesis_id.as_str(), "reference_hypothesis_id"),
                (case.observed_outcome_id.as_str(), "observed_outcome_id"),
                (case.reference_label_evidence_ref.as_str(), "reference_label_evidence_ref"),
                (case.observation_evidence_ref.as_str(), "observation_evidence_ref"),
            ] {
                if !nonempty(value) {
                    return Err(CalibrationError::EmptyField {
                        case_id: case.case_id.clone(),
                        field,
                    });
                }
            }

            if !case_ids.insert(case.case_id.as_str()) {
                return Err(CalibrationError::DuplicateCaseId(case.case_id.clone()));
            }
            if !acquisition_ids.insert(case.acquisition_lineage_id.as_str()) {
                return Err(CalibrationError::DuplicateAcquisitionLineage(
                    case.acquisition_lineage_id.clone(),
                ));
            }

            match site_splits.get(case.site_id.as_str()) {
                Some(existing) if *existing != case.split => {
                    return Err(CalibrationError::SiteAppearsInMultipleSplits(
                        case.site_id.clone(),
                    ));
                }
                Some(_) => {}
                None => {
                    site_splits.insert(case.site_id.as_str(), case.split);
                }
            }

            if !site_actions.insert((case.site_id.as_str(), case.action_id.as_str())) {
                return Err(CalibrationError::DuplicateSiteAction {
                    site_id: case.site_id.clone(),
                    action_id: case.action_id.clone(),
                });
            }

            let Some(outcomes) = actions.get(case.action_id.as_str()) else {
                return Err(CalibrationError::UnknownAction {
                    case_id: case.case_id.clone(),
                    action_id: case.action_id.clone(),
                });
            };
            if !hypothesis_ids.contains(case.reference_hypothesis_id.as_str()) {
                return Err(CalibrationError::UnknownHypothesis {
                    case_id: case.case_id.clone(),
                    hypothesis_id: case.reference_hypothesis_id.clone(),
                });
            }
            if !outcomes.contains(&case.observed_outcome_id) {
                return Err(CalibrationError::UnknownOutcome {
                    case_id: case.case_id.clone(),
                    action_id: case.action_id.clone(),
                    outcome_id: case.observed_outcome_id.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn fit_likelihoods(
        &self,
        dirichlet_alpha: f64,
        minimum_training_sites: usize,
    ) -> Result<EmpiricalLikelihoodTable, CalibrationError> {
        self.validate()?;
        if !dirichlet_alpha.is_finite() || dirichlet_alpha <= 0.0 {
            return Err(CalibrationError::InvalidDirichletAlpha);
        }
        if minimum_training_sites == 0 {
            return Err(CalibrationError::InvalidMinimumTrainingSites);
        }

        let training_cases = self
            .cases
            .iter()
            .filter(|case| case.split == DataSplit::Training)
            .collect::<Vec<_>>();
        if training_cases.is_empty() {
            return Err(CalibrationError::NoTrainingCases);
        }

        let mut counts = BTreeMap::<(String, String, String), usize>::new();
        let mut totals = BTreeMap::<(String, String), usize>::new();
        for case in training_cases {
            *counts
                .entry((
                    case.action_id.clone(),
                    case.reference_hypothesis_id.clone(),
                    case.observed_outcome_id.clone(),
                ))
                .or_default() += 1;
            *totals
                .entry((
                    case.action_id.clone(),
                    case.reference_hypothesis_id.clone(),
                ))
                .or_default() += 1;
        }

        let mut cells = Vec::new();
        for action in &self.actions {
            for hypothesis_id in &self.hypothesis_ids {
                let key = (action.action_id.clone(), hypothesis_id.clone());
                let sample_count = totals.get(&key).copied().unwrap_or(0);

                let probabilities = if sample_count >= minimum_training_sites {
                    let denominator = sample_count as f64
                        + dirichlet_alpha * action.outcome_ids.len() as f64;
                    if !denominator.is_finite() || denominator <= 0.0 {
                        return Err(CalibrationError::NumericalOverflow(
                            "Dirichlet denominator",
                        ));
                    }
                    let values = action
                        .outcome_ids
                        .iter()
                        .map(|outcome_id| {
                            let count = counts
                                .get(&(
                                    action.action_id.clone(),
                                    hypothesis_id.clone(),
                                    outcome_id.clone(),
                                ))
                                .copied()
                                .unwrap_or(0);
                            (count as f64 + dirichlet_alpha) / denominator
                        })
                        .collect::<Vec<_>>();
                    let sum = values.iter().sum::<f64>();
                    if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-9 {
                        return Err(CalibrationError::NumericalOverflow(
                            "Dirichlet predictive normalization",
                        ));
                    }
                    Some(values)
                } else {
                    None
                };

                cells.push(LikelihoodCell {
                    action_id: action.action_id.clone(),
                    hypothesis_id: hypothesis_id.clone(),
                    outcome_ids: action.outcome_ids.clone(),
                    training_sample_count: sample_count,
                    probabilities,
                });
            }
        }

        cells.sort_by(|left, right| {
            left.action_id
                .cmp(&right.action_id)
                .then_with(|| left.hypothesis_id.cmp(&right.hypothesis_id))
        });

        Ok(EmpiricalLikelihoodTable {
            dataset_id: self.dataset_id.clone(),
            hypothesis_ids: self.hypothesis_ids.clone(),
            action_ids: self.actions.iter().map(|action| action.action_id.clone()).collect(),
            action_modalities: self
                .actions
                .iter()
                .map(|action| (action.action_id.clone(), action.modality.clone()))
                .collect(),
            estimator_id: "dirichlet-categorical-v0.1".to_string(),
            training_split_only: true,
            dirichlet_alpha,
            minimum_training_sites,
            cells,
        })
    }

    pub fn score_holdout(
        &self,
        table: &EmpiricalLikelihoodTable,
        split: DataSplit,
    ) -> Result<HoldoutScoreReport, CalibrationError> {
        self.validate()?;
        if split == DataSplit::Training {
            return Err(CalibrationError::CannotScoreTrainingSplit);
        }
        if table.dataset_id != self.dataset_id
            || !table.training_split_only
            || table.estimator_id != "dirichlet-categorical-v0.1"
            || table.hypothesis_ids != self.hypothesis_ids
            || table.action_ids != self.actions.iter().map(|action| action.action_id.clone()).collect::<Vec<_>>()
            || table.action_modalities
                != self.actions.iter().map(|action| (action.action_id.clone(), action.modality.clone())).collect::<Vec<_>>()
            || !table.dirichlet_alpha.is_finite()
            || table.dirichlet_alpha <= 0.0
            || table.minimum_training_sites == 0
        {
            return Err(CalibrationError::InvalidLikelihoodTable);
        }
        let expected_cells = self.actions.len() * self.hypothesis_ids.len();
        if table.cells.len() != expected_cells {
            return Err(CalibrationError::InvalidLikelihoodTable);
        }
        let expected_table = self.fit_likelihoods(
            table.dirichlet_alpha,
            table.minimum_training_sites,
        )?;
        if expected_table.cells != table.cells {
            return Err(CalibrationError::InvalidLikelihoodTable);
        }
        let mut seen_cells = BTreeSet::new();
        for cell in &table.cells {
            let Some(action) = self.actions.iter().find(|a| a.action_id == cell.action_id) else {
                return Err(CalibrationError::InvalidLikelihoodTable);
            };
            if !self.hypothesis_ids.contains(&cell.hypothesis_id)
                || cell.outcome_ids != action.outcome_ids
                || !seen_cells.insert((cell.action_id.as_str(), cell.hypothesis_id.as_str()))
            {
                return Err(CalibrationError::InvalidLikelihoodTable);
            }
            match &cell.probabilities {
                Some(probabilities) => {
                    let sum = probabilities.iter().sum::<f64>();
                    if probabilities.len() != cell.outcome_ids.len()
                        || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                        || !sum.is_finite()
                        || (sum - 1.0).abs() > 1.0e-9
                        || cell.training_sample_count < table.minimum_training_sites
                    {
                        return Err(CalibrationError::InvalidLikelihoodTable);
                    }
                }
                None if cell.training_sample_count < table.minimum_training_sites => {}
                None => return Err(CalibrationError::InvalidLikelihoodTable),
            }
        }

        let cases = self
            .cases
            .iter()
            .filter(|case| case.split == split)
            .collect::<Vec<_>>();
        if cases.is_empty() {
            return Err(CalibrationError::NoCasesForSplit(split));
        }

        let mut scores = BTreeMap::<(String, String), Vec<(f64, f64)>>::new();
        let mut unscored = 0usize;

        for case in &cases {
            let cell = table.cells.iter().find(|cell| {
                cell.action_id == case.action_id
                    && cell.hypothesis_id == case.reference_hypothesis_id
            });
            let Some(cell) = cell else {
                unscored += 1;
                continue;
            };
            let Some(probabilities) = &cell.probabilities else {
                unscored += 1;
                continue;
            };
            let Some(outcome_index) = cell
                .outcome_ids
                .iter()
                .position(|outcome_id| outcome_id == &case.observed_outcome_id)
            else {
                unscored += 1;
                continue;
            };

            let observed_probability = probabilities[outcome_index];
            if !observed_probability.is_finite() || observed_probability <= 0.0 {
                return Err(CalibrationError::NumericalOverflow(
                    "observed outcome probability",
                ));
            }
            let nll = -observed_probability.ln();
            let brier = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| {
                    let target = if index == outcome_index { 1.0 } else { 0.0 };
                    (probability - target).powi(2)
                })
                .sum::<f64>();
            if !nll.is_finite() || !brier.is_finite() {
                return Err(CalibrationError::NumericalOverflow("holdout scoring"));
            }

            scores
                .entry((case.action_id.clone(), case.reference_hypothesis_id.clone()))
                .or_default()
                .push((brier, nll));
        }

        let scored_cases = scores.values().map(Vec::len).sum::<usize>();
        if scored_cases == 0 {
            return Err(CalibrationError::NoScorableCases(split));
        }

        let all_scores = scores.values().flatten().copied().collect::<Vec<_>>();
        let mean_brier = all_scores.iter().map(|(brier, _)| brier).sum::<f64>()
            / all_scores.len() as f64;
        let mean_nll =
            all_scores.iter().map(|(_, nll)| nll).sum::<f64>() / all_scores.len() as f64;

        let mut cell_reports = scores
            .into_iter()
            .map(|((action_id, hypothesis_id), entries)| {
                let count = entries.len();
                CellHoldoutScore {
                    action_id,
                    hypothesis_id,
                    sample_count: count,
                    mean_brier: entries.iter().map(|(brier, _)| brier).sum::<f64>()
                        / count as f64,
                    mean_negative_log_likelihood: entries
                        .iter()
                        .map(|(_, nll)| nll)
                        .sum::<f64>()
                        / count as f64,
                }
            })
            .collect::<Vec<_>>();
        cell_reports.sort_by(|left, right| {
            left.action_id
                .cmp(&right.action_id)
                .then_with(|| left.hypothesis_id.cmp(&right.hypothesis_id))
        });

        if !mean_brier.is_finite() || !mean_nll.is_finite() {
            return Err(CalibrationError::NumericalOverflow("mean holdout scores"));
        }

        Ok(HoldoutScoreReport {
            split,
            total_cases: cases.len(),
            scored_cases,
            unscored_cases: unscored,
            coverage_fraction: scored_cases as f64 / cases.len() as f64,
            mean_brier,
            mean_negative_log_likelihood: mean_nll,
            cells: cell_reports,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LikelihoodCell {
    pub action_id: String,
    pub hypothesis_id: String,
    pub outcome_ids: Vec<String>,
    pub training_sample_count: usize,
    /// None means insufficient independent training-site support; do not pass
    /// this cell to an EVSI planner as an invented distribution.
    pub probabilities: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalLikelihoodTable {
    pub dataset_id: String,
    pub hypothesis_ids: Vec<String>,
    pub action_ids: Vec<String>,
    pub action_modalities: Vec<(String, String)>,
    pub estimator_id: String,
    pub training_split_only: bool,
    pub dirichlet_alpha: f64,
    pub minimum_training_sites: usize,
    pub cells: Vec<LikelihoodCell>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementActionPolicy {
    pub action_id: String,
    pub resource_cost: ResourceCost,
    /// Supplied by the separate reviewed mission/ODD/preservation gate.
    pub external_gate_passed: bool,
}

impl EmpiricalLikelihoodTable {
    pub fn cell(&self, action_id: &str, hypothesis_id: &str) -> Option<&LikelihoodCell> {
        self.cells.iter().find(|cell| {
            cell.action_id == action_id && cell.hypothesis_id == hypothesis_id
        })
    }

    /// Build the one-step EVSI input without inventing priors, losses, resource
    /// costs or permission. The caller supplies those independently. Every
    /// action/hypothesis likelihood cell must have sufficient Training support.
    pub fn to_planning_problem(
        &self,
        hypotheses: Vec<PlannerHypothesis>,
        decisions: Vec<PlannerMissionDecision>,
        action_policies: Vec<MeasurementActionPolicy>,
    ) -> Result<PlanningProblem, CalibrationError> {
        let supplied_hypothesis_ids = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.id.clone())
            .collect::<Vec<_>>();
        if supplied_hypothesis_ids != self.hypothesis_ids {
            return Err(CalibrationError::PlannerHypothesisOrderMismatch);
        }
        if decisions
            .iter()
            .any(|decision| decision.loss_by_hypothesis.len() != hypotheses.len())
        {
            return Err(CalibrationError::PlannerDecisionDimensionMismatch);
        }

        let mut policy_map = BTreeMap::new();
        for policy in action_policies {
            let id = policy.action_id.clone();
            if policy_map.insert(id.clone(), policy).is_some() {
                return Err(CalibrationError::DuplicatePlannerActionPolicy(id));
            }
        }
        if policy_map.len() != self.action_ids.len()
            || self.action_ids.iter().any(|id| !policy_map.contains_key(id))
        {
            return Err(CalibrationError::PlannerActionPolicyMismatch);
        }

        let mut actions = Vec::with_capacity(self.action_ids.len());
        for action_id in &self.action_ids {
            let first_hypothesis = self
                .hypothesis_ids
                .first()
                .ok_or(CalibrationError::InvalidLikelihoodTable)?;
            let first_cell = self
                .cell(action_id, first_hypothesis)
                .ok_or(CalibrationError::InvalidLikelihoodTable)?;
            let outcome_ids = first_cell.outcome_ids.clone();

            let mut probabilities_by_hypothesis = Vec::with_capacity(self.hypothesis_ids.len());
            for hypothesis_id in &self.hypothesis_ids {
                let cell = self
                    .cell(action_id, hypothesis_id)
                    .ok_or(CalibrationError::InvalidLikelihoodTable)?;
                if cell.outcome_ids != outcome_ids {
                    return Err(CalibrationError::InvalidLikelihoodTable);
                }
                let Some(probabilities) = &cell.probabilities else {
                    return Err(CalibrationError::InsufficientLikelihoodSupport {
                        action_id: action_id.clone(),
                        hypothesis_id: hypothesis_id.clone(),
                        training_sample_count: cell.training_sample_count,
                    });
                };
                if probabilities.len() != outcome_ids.len() {
                    return Err(CalibrationError::InvalidLikelihoodTable);
                }
                probabilities_by_hypothesis.push(probabilities.clone());
            }

            let outcomes = outcome_ids
                .iter()
                .enumerate()
                .map(|(outcome_index, outcome_id)| PlannerObservationOutcome {
                    id: outcome_id.clone(),
                    likelihood_by_hypothesis: probabilities_by_hypothesis
                        .iter()
                        .map(|probabilities| probabilities[outcome_index])
                        .collect(),
                })
                .collect();

            let policy = policy_map
                .get(action_id)
                .ok_or(CalibrationError::PlannerActionPolicyMismatch)?;
            let modality = self
                .action_modalities
                .iter()
                .find(|(id, _)| id == action_id)
                .map(|(_, modality)| modality.clone())
                .ok_or(CalibrationError::InvalidLikelihoodTable)?;
            actions.push(PlannerMeasurementAction {
                id: action_id.clone(),
                modality,
                outcomes,
                cost: policy.resource_cost,
                external_gate_passed: policy.external_gate_passed,
            });
        }

        Ok(PlanningProblem {
            hypotheses,
            decisions,
            actions,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellHoldoutScore {
    pub action_id: String,
    pub hypothesis_id: String,
    pub sample_count: usize,
    pub mean_brier: f64,
    pub mean_negative_log_likelihood: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutScoreReport {
    pub split: DataSplit,
    pub total_cases: usize,
    pub scored_cases: usize,
    pub unscored_cases: usize,
    pub coverage_fraction: f64,
    pub mean_brier: f64,
    pub mean_negative_log_likelihood: f64,
    pub cells: Vec<CellHoldoutScore>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(
        id: &str,
        site: &str,
        action: &str,
        hypothesis: &str,
        outcome: &str,
        split: DataSplit,
    ) -> CalibrationCase {
        CalibrationCase {
            case_id: id.into(),
            site_id: site.into(),
            acquisition_lineage_id: format!("acq-{id}"),
            action_id: action.into(),
            reference_hypothesis_id: hypothesis.into(),
            observed_outcome_id: outcome.into(),
            split,
            reference_label_evidence_ref: format!("label-ref-{id}"),
            observation_evidence_ref: format!("obs-ref-{id}"),
        }
    }

    fn dataset() -> CalibrationDataset {
        CalibrationDataset {
            dataset_id: "synthetic-site-disjoint-v1".into(),
            hypothesis_ids: vec![
                "tunnel".into(),
                "collapse".into(),
                "unknown".into(),
            ],
            actions: vec![ActionContract {
                action_id: "gpr-survey-01".into(),
                modality: "ground_penetrating_radar".into(),
                outcome_ids: vec!["signal".into(), "no_signal".into()],
            }],
            cases: vec![
                case("t1", "site-t1", "gpr-survey-01", "tunnel", "signal", DataSplit::Training),
                case("t2", "site-t2", "gpr", "tunnel", "signal", DataSplit::Training),
                case("t3", "site-t3", "gpr", "tunnel", "no_signal", DataSplit::Training),
                case("c1", "site-c1", "gpr", "collapse", "no_signal", DataSplit::Training),
                case("c2", "site-c2", "gpr", "collapse", "no_signal", DataSplit::Training),
                case("u1", "site-u1", "gpr", "unknown", "signal", DataSplit::Training),
                case("u2", "site-u2", "gpr", "unknown", "no_signal", DataSplit::Training),
                case("cal1", "site-cal1", "gpr", "tunnel", "signal", DataSplit::Calibration),
                case("cal2", "site-cal2", "gpr", "collapse", "no_signal", DataSplit::Calibration),
                case("eval1", "site-eval1", "gpr", "tunnel", "signal", DataSplit::Evaluation),
                case("eval2", "site-eval2", "gpr", "collapse", "no_signal", DataSplit::Evaluation),
            ],
        }
    }

    #[test]
    fn unknown_hypothesis_is_mandatory() {
        let mut data = dataset();
        data.hypothesis_ids.retain(|id| id != UNKNOWN_HYPOTHESIS_ID);
        assert_eq!(
            data.validate(),
            Err(CalibrationError::MissingUnknownHypothesis)
        );
    }

    #[test]
    fn a_site_cannot_leak_across_splits() {
        let mut data = dataset();
        data.cases.push(case(
            "leak",
            "site-t1",
            "gpr",
            "tunnel",
            "signal",
            DataSplit::Evaluation,
        ));
        assert_eq!(
            data.validate(),
            Err(CalibrationError::SiteAppearsInMultipleSplits("site-t1".into()))
        );
    }

    #[test]
    fn one_acquisition_cannot_be_counted_twice() {
        let mut data = dataset();
        data.cases[1].acquisition_lineage_id =
            data.cases[0].acquisition_lineage_id.clone();
        assert_eq!(
            data.validate(),
            Err(CalibrationError::DuplicateAcquisitionLineage(
                "acq-t1".into()
            ))
        );
    }

    #[test]
    fn one_site_action_case_is_allowed_at_most_once() {
        let mut data = dataset();
        let mut duplicate = data.cases[0].clone();
        duplicate.case_id = "second-tunnel-case".into();
        duplicate.acquisition_lineage_id = "acq-second".into();
        data.cases.push(duplicate);
        assert_eq!(
            data.validate(),
            Err(CalibrationError::DuplicateSiteAction {
                site_id: "site-t1".into(),
                action_id: "gpr".into()
            })
        );
    }

    #[test]
    fn duplicate_site_hypothesis_label_is_not_silently_double_counted() {
        let mut data = dataset();
        data.cases[0].case_id = data.cases[1].case_id.clone();
        assert_eq!(
            data.validate(),
            Err(CalibrationError::DuplicateCaseId("t2".into()))
        );
    }

    #[test]
    fn dirichlet_likelihoods_use_training_split_only() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 3).unwrap();
        let tunnel = table.cell("gpr", "tunnel").unwrap();
        assert_eq!(tunnel.training_sample_count, 3);
        let probabilities = tunnel.probabilities.as_ref().unwrap();
        assert!((probabilities[0] - 0.625).abs() < 1.0e-12);
        assert!((probabilities[1] - 0.375).abs() < 1.0e-12);
        assert!(table.training_split_only);
    }

    #[test]
    fn under_supported_unknown_cell_has_no_fabricated_likelihood() {
        let mut data = dataset();
        data.cases.retain(|case| {
            case.reference_hypothesis_id != "unknown"
                || case.split != DataSplit::Training
        });
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let unknown = table.cell("gpr", "unknown").unwrap();
        assert_eq!(unknown.training_sample_count, 0);
        assert!(unknown.probabilities.is_none());
    }

    #[test]
    fn holdout_scoring_reports_brier_and_nll() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let report = data.score_holdout(&table, DataSplit::Evaluation).unwrap();
        assert_eq!(report.total_cases, 2);
        assert_eq!(report.scored_cases, 2);
        assert_eq!(report.unscored_cases, 0);
        assert_eq!(report.coverage_fraction, 1.0);
        assert!(report.mean_brier.is_finite() && report.mean_brier >= 0.0);
        assert!(report.mean_negative_log_likelihood.is_finite());
    }

    #[test]
    fn insufficient_training_support_reduces_holdout_coverage() {
        let mut data = dataset();
        data.cases.retain(|case| case.case_id != "u2");
        data.cases.push(case(
            "eval-unknown",
            "site-eval-unknown",
            "gpr",
            "unknown",
            "no_signal",
            DataSplit::Evaluation,
        ));
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let report = data.score_holdout(&table, DataSplit::Evaluation).unwrap();
        assert_eq!(report.total_cases, 3);
        assert_eq!(report.scored_cases, 2);
        assert_eq!(report.unscored_cases, 1);
        assert!((report.coverage_fraction - (2.0 / 3.0)).abs() < 1.0e-12);
    }

    fn planner_hypotheses() -> Vec<PlannerHypothesis> {
        vec![
            PlannerHypothesis {
                id: "tunnel".into(),
                prior_probability: 0.45,
            },
            PlannerHypothesis {
                id: "collapse".into(),
                prior_probability: 0.45,
            },
            PlannerHypothesis {
                id: "unknown".into(),
                prior_probability: 0.10,
            },
        ]
    }

    fn planner_decisions() -> Vec<PlannerMissionDecision> {
        vec![
            PlannerMissionDecision {
                id: "enter".into(),
                loss_by_hypothesis: vec![0.0, 100.0, 50.0],
            },
            PlannerMissionDecision {
                id: "hold".into(),
                loss_by_hypothesis: vec![30.0, 0.0, 10.0],
            },
        ]
    }

    fn planner_policy(allowed: bool) -> MeasurementActionPolicy {
        MeasurementActionPolicy {
            action_id: "gpr".into(),
            resource_cost: ResourceCost {
                energy_j: 100.0,
                duration_s: 10.0,
                data_bytes: 1_000.0,
                exposure_units: 0.1,
            },
            external_gate_passed: allowed,
        }
    }

    #[test]
    fn calibrated_table_converts_to_valid_evsi_problem() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let problem = table
            .to_planning_problem(
                planner_hypotheses(),
                planner_decisions(),
                vec![planner_policy(true)],
            )
            .unwrap();
        assert!(problem.validate().is_ok());
        let signal = problem.actions[0]
            .outcomes
            .iter()
            .find(|outcome| outcome.id == "signal")
            .unwrap();
        assert!((signal.likelihood_by_hypothesis[0] - 0.625).abs() < 1.0e-12);
        assert!((signal.likelihood_by_hypothesis[1] - (0.5 / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn planner_conversion_preserves_external_gate_denial() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let problem = table
            .to_planning_problem(
                planner_hypotheses(),
                planner_decisions(),
                vec![planner_policy(false)],
            )
            .unwrap();
        assert!(!problem.actions[0].external_gate_passed);
    }

    #[test]
    fn unsupported_cells_cannot_be_converted_to_evsi_probabilities() {
        let mut data = dataset();
        data.cases.retain(|case| case.case_id != "u2");
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        assert_eq!(
            table.to_planning_problem(
                planner_hypotheses(),
                planner_decisions(),
                vec![planner_policy(true)],
            ),
            Err(CalibrationError::InsufficientLikelihoodSupport {
                action_id: "gpr".into(),
                hypothesis_id: "unknown".into(),
                training_sample_count: 1,
            })
        );
    }

    #[test]
    fn planner_conversion_rejects_hypothesis_order_drift() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let mut hypotheses = planner_hypotheses();
        hypotheses.swap(0, 1);
        assert_eq!(
            table.to_planning_problem(
                hypotheses,
                planner_decisions(),
                vec![planner_policy(true)],
            ),
            Err(CalibrationError::PlannerHypothesisOrderMismatch)
        );
    }

    #[test]
    fn planner_conversion_requires_exact_action_policy_set() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        assert_eq!(
            table.to_planning_problem(
                planner_hypotheses(),
                planner_decisions(),
                Vec::new(),
            ),
            Err(CalibrationError::PlannerActionPolicyMismatch)
        );
    }

    #[test]
    fn altered_likelihood_table_is_rejected_before_holdout_scoring() {
        let data = dataset();
        let mut table = data.fit_likelihoods(0.5, 2).unwrap();
        table
            .cells
            .iter_mut()
            .find(|cell| cell.action_id == "gpr" && cell.hypothesis_id == "tunnel")
            .unwrap()
            .probabilities
            .as_mut()
            .unwrap()[0] = 0.99;
        assert_eq!(
            data.score_holdout(&table, DataSplit::Evaluation),
            Err(CalibrationError::InvalidLikelihoodTable)
        );
    }

    #[test]
    fn training_split_cannot_be_misreported_as_holdout() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        assert_eq!(
            data.score_holdout(&table, DataSplit::Training),
            Err(CalibrationError::CannotScoreTrainingSplit)
        );
    }

    #[test]
    fn calibration_and_evaluation_splits_are_scored_separately() {
        let data = dataset();
        let table = data.fit_likelihoods(0.5, 2).unwrap();
        let calibration = data.score_holdout(&table, DataSplit::Calibration).unwrap();
        let evaluation = data.score_holdout(&table, DataSplit::Evaluation).unwrap();
        assert_eq!(calibration.split, DataSplit::Calibration);
        assert_eq!(evaluation.split, DataSplit::Evaluation);
        assert_eq!(calibration.total_cases, 2);
        assert_eq!(evaluation.total_cases, 2);
    }

    #[test]
    fn invalid_smoothing_or_support_threshold_is_rejected() {
        let data = dataset();
        assert_eq!(
            data.fit_likelihoods(0.0, 1),
            Err(CalibrationError::InvalidDirichletAlpha)
        );
        assert_eq!(
            data.fit_likelihoods(0.5, 0),
            Err(CalibrationError::InvalidMinimumTrainingSites)
        );
    }

    #[test]
    fn action_contract_must_cover_every_case_outcome() {
        let mut data = dataset();
        data.cases[0].observed_outcome_id = "unknown-outcome".into();
        assert!(matches!(
            data.validate(),
            Err(CalibrationError::UnknownOutcome { .. })
        ));
    }
}
