//! Evidence contracts for next-best-observation planning.
//!
//! This crate does not choose an observation by hidden utility scalarization.
//! It records expected information gain, uncertainty reduction, cost, delay,
//! energy, data volume, human effort, operational risk, and intrusiveness as
//! separate dimensions. A conservative Pareto frontier may remove candidates
//! that are clearly dominated, but final selection remains an explicit policy
//! or human decision outside this crate.

use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_earth_observation::{Hypothesis, SensorModality};

pub type Result<T> = std::result::Result<T, PlanningError>;

#[derive(Debug, Clone, PartialEq)]
pub enum PlanningError {
    EmptyField(&'static str),
    NonFinite { field: &'static str, value: f64 },
    Negative { field: &'static str, value: f64 },
    InvalidProbability(f64),
    InvalidReduction(f64),
    MissingHypothesis,
    MissingModel,
    DuplicateCandidate(String),
}

impl Display for PlanningError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::NonFinite { field, value } => write!(f, "{field} must be finite, got {value}"),
            Self::Negative { field, value } => write!(f, "{field} must be non-negative, got {value}"),
            Self::InvalidProbability(value) => {
                write!(f, "probability/risk must be in [0, 1], got {value}")
            }
            Self::InvalidReduction(value) => {
                write!(f, "uncertainty reduction must be in [0, 1], got {value}")
            }
            Self::MissingHypothesis => write!(f, "an observation candidate must target at least one hypothesis"),
            Self::MissingModel => write!(f, "candidate evaluation requires a planner/model reference"),
            Self::DuplicateCandidate(id) => write!(f, "duplicate observation candidate id {id}"),
        }
    }
}

impl Error for PlanningError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(PlanningError::EmptyField(field));
    }
    Ok(())
}

fn non_negative(value: f64, field: &'static str) -> Result<f64> {
    if !value.is_finite() {
        return Err(PlanningError::NonFinite { field, value });
    }
    if value < 0.0 {
        return Err(PlanningError::Negative { field, value });
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationActionKind {
    ReuseExistingData,
    SatelliteAcquisition,
    AirborneSurvey,
    GroundSurvey,
    InSituMeasurement,
    LaboratoryAnalysis,
    BoreholeOrIntrusiveSurvey,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObservationIntrusiveness {
    NonInvasive,
    MinimallyInvasive,
    Invasive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisRef {
    pub id: String,
    pub statement: String,
}

impl TryFrom<&Hypothesis> for HypothesisRef {
    type Error = PlanningError;

    fn try_from(value: &Hypothesis) -> Result<Self> {
        non_empty(&value.id, "hypothesis id")?;
        non_empty(&value.statement, "hypothesis statement")?;
        Ok(Self {
            id: value.id.clone(),
            statement: value.statement.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerModelRef {
    pub model_id: String,
    pub version: String,
    pub artifact_digest: Option<String>,
}

impl PlannerModelRef {
    pub fn new(model_id: impl Into<String>, version: impl Into<String>) -> Result<Self> {
        let model_id = model_id.into();
        let version = version.into();
        non_empty(&model_id, "planner model id")?;
        non_empty(&version, "planner model version")?;
        Ok(Self {
            model_id,
            version,
            artifact_digest: None,
        })
    }

    pub fn with_artifact_digest(mut self, digest: impl Into<String>) -> Result<Self> {
        let digest = digest.into();
        non_empty(&digest, "planner model artifact digest")?;
        self.artifact_digest = Some(digest);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationCandidate {
    pub id: String,
    pub description: String,
    pub action_kind: ObservationActionKind,
    pub modality: SensorModality,
    pub target_hypothesis_ids: Vec<String>,
    pub intrusiveness: ObservationIntrusiveness,
}

impl ObservationCandidate {
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        action_kind: ObservationActionKind,
        modality: SensorModality,
        target_hypothesis_ids: Vec<String>,
        intrusiveness: ObservationIntrusiveness,
    ) -> Result<Self> {
        let id = id.into();
        let description = description.into();
        non_empty(&id, "observation candidate id")?;
        non_empty(&description, "observation candidate description")?;
        if target_hypothesis_ids.is_empty() {
            return Err(PlanningError::MissingHypothesis);
        }
        for hypothesis_id in &target_hypothesis_ids {
            non_empty(hypothesis_id, "target hypothesis id")?;
        }
        Ok(Self {
            id,
            description,
            action_kind,
            modality,
            target_hypothesis_ids,
            intrusiveness,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationCostVector {
    pub delay_seconds: Option<f64>,
    pub energy_joules: Option<f64>,
    pub downlink_bytes: Option<f64>,
    pub human_hours: Option<f64>,
    pub monetary_cost: Option<(f64, String)>,
    pub operational_risk: Option<f64>,
}

impl ObservationCostVector {
    pub const fn unknown() -> Self {
        Self {
            delay_seconds: None,
            energy_joules: None,
            downlink_bytes: None,
            human_hours: None,
            monetary_cost: None,
            operational_risk: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("delay seconds", self.delay_seconds),
            ("energy joules", self.energy_joules),
            ("downlink bytes", self.downlink_bytes),
            ("human hours", self.human_hours),
        ] {
            if let Some(value) = value {
                non_negative(value, field)?;
            }
        }
        if let Some((amount, currency)) = &self.monetary_cost {
            non_negative(*amount, "monetary cost")?;
            non_empty(currency, "monetary cost currency")?;
        }
        if let Some(risk) = self.operational_risk {
            if !risk.is_finite() || !(0.0..=1.0).contains(&risk) {
                return Err(PlanningError::InvalidProbability(risk));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisInformationGain {
    pub hypothesis_id: String,
    /// Expected fractional reduction in uncertainty for this hypothesis.
    pub expected_uncertainty_reduction: f64,
}

impl HypothesisInformationGain {
    pub fn new(
        hypothesis_id: impl Into<String>,
        expected_uncertainty_reduction: f64,
    ) -> Result<Self> {
        let hypothesis_id = hypothesis_id.into();
        non_empty(&hypothesis_id, "hypothesis id")?;
        if !expected_uncertainty_reduction.is_finite()
            || !(0.0..=1.0).contains(&expected_uncertainty_reduction)
        {
            return Err(PlanningError::InvalidReduction(
                expected_uncertainty_reduction,
            ));
        }
        Ok(Self {
            hypothesis_id,
            expected_uncertainty_reduction,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationCandidateEvaluation {
    pub candidate: ObservationCandidate,
    /// Expected information gain in nats. This is not a utility score.
    pub expected_information_gain_nats: f64,
    pub hypothesis_information_gain: Vec<HypothesisInformationGain>,
    pub costs: ObservationCostVector,
    pub model: PlannerModelRef,
    pub assumptions: Vec<String>,
}

impl ObservationCandidateEvaluation {
    pub fn new(
        candidate: ObservationCandidate,
        expected_information_gain_nats: f64,
        hypothesis_information_gain: Vec<HypothesisInformationGain>,
        costs: ObservationCostVector,
        model: PlannerModelRef,
        assumptions: Vec<String>,
    ) -> Result<Self> {
        non_negative(
            expected_information_gain_nats,
            "expected information gain nats",
        )?;
        costs.validate()?;
        non_empty(&model.model_id, "planner model id")?;
        for assumption in &assumptions {
            non_empty(assumption, "planner assumption")?;
        }
        Ok(Self {
            candidate,
            expected_information_gain_nats,
            hypothesis_information_gain,
            costs,
            model,
            assumptions,
        })
    }
}

/// A collection of candidate observation evaluations.
///
/// `pareto_frontier` is deliberately conservative: candidate A can dominate B
/// only when every cost dimension needed for the comparison is known for both.
/// Missing costs never count as zero.
#[derive(Debug, Clone, Default)]
pub struct ObservationPlan {
    candidates: Vec<ObservationCandidateEvaluation>,
}

impl ObservationPlan {
    pub fn new(candidates: Vec<ObservationCandidateEvaluation>) -> Result<Self> {
        let mut ids = std::collections::HashSet::new();
        for candidate in &candidates {
            if !ids.insert(candidate.candidate.id.clone()) {
                return Err(PlanningError::DuplicateCandidate(
                    candidate.candidate.id.clone(),
                ));
            }
        }
        Ok(Self { candidates })
    }

    pub fn candidates(&self) -> &[ObservationCandidateEvaluation] {
        &self.candidates
    }

    pub fn pareto_frontier(&self) -> Vec<&ObservationCandidateEvaluation> {
        self.candidates
            .iter()
            .filter(|candidate| {
                !self
                    .candidates
                    .iter()
                    .any(|other| other.candidate.id != candidate.candidate.id && dominates(other, candidate))
            })
            .collect()
    }
}

fn lower_or_equal(a: Option<f64>, b: Option<f64>) -> Option<bool> {
    Some(a? <= b?)
}

fn monetary_lower_or_equal(
    a: &Option<(f64, String)>,
    b: &Option<(f64, String)>,
) -> Option<bool> {
    let (a_amount, a_currency) = a.as_ref()?;
    let (b_amount, b_currency) = b.as_ref()?;
    if a_currency != b_currency {
        return None;
    }
    Some(a_amount <= b_amount)
}

fn dominates(
    a: &ObservationCandidateEvaluation,
    b: &ObservationCandidateEvaluation,
) -> bool {
    if a.expected_information_gain_nats < b.expected_information_gain_nats {
        return false;
    }
    if a.candidate.intrusiveness > b.candidate.intrusiveness {
        return false;
    }

    let comparisons = [
        lower_or_equal(a.costs.delay_seconds, b.costs.delay_seconds),
        lower_or_equal(a.costs.energy_joules, b.costs.energy_joules),
        lower_or_equal(a.costs.downlink_bytes, b.costs.downlink_bytes),
        lower_or_equal(a.costs.human_hours, b.costs.human_hours),
        lower_or_equal(a.costs.operational_risk, b.costs.operational_risk),
        monetary_lower_or_equal(&a.costs.monetary_cost, &b.costs.monetary_cost),
    ];

    // Conservative rule: if a cost dimension is unknown or incomparable, no
    // automatic dominance claim is made.
    if comparisons.iter().any(Option::is_none) {
        return false;
    }
    if comparisons.iter().any(|result| result == &Some(false)) {
        return false;
    }

    a.expected_information_gain_nats > b.expected_information_gain_nats
        || a.candidate.intrusiveness < b.candidate.intrusiveness
        || a.costs.delay_seconds < b.costs.delay_seconds
        || a.costs.energy_joules < b.costs.energy_joules
        || a.costs.downlink_bytes < b.costs.downlink_bytes
        || a.costs.human_hours < b.costs.human_hours
        || a.costs.operational_risk < b.costs.operational_risk
        || a.costs.monetary_cost.as_ref().map(|v| v.0)
            < b.costs.monetary_cost.as_ref().map(|v| v.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_earth_observation::{RadarBand, Polarization};

    fn sar() -> SensorModality {
        SensorModality::SyntheticApertureRadar {
            band: RadarBand::L,
            polarization: Some(Polarization::Vv),
        }
    }

    fn model() -> PlannerModelRef {
        PlannerModelRef::new("fixture-planner", "0.1.0").unwrap()
    }

    fn candidate(id: &str, intrusiveness: ObservationIntrusiveness) -> ObservationCandidate {
        ObservationCandidate::new(
            id,
            format!("candidate {id}"),
            ObservationActionKind::SatelliteAcquisition,
            sar(),
            vec!["cavity".into(), "groundwater".into()],
            intrusiveness,
        )
        .unwrap()
    }

    fn known_cost(delay: f64, energy: f64, risk: f64) -> ObservationCostVector {
        ObservationCostVector {
            delay_seconds: Some(delay),
            energy_joules: Some(energy),
            downlink_bytes: Some(10_000.0),
            human_hours: Some(1.0),
            monetary_cost: Some((100.0, "USD".into())),
            operational_risk: Some(risk),
        }
    }

    #[test]
    fn information_gain_is_not_a_permission_to_execute() {
        let evaluation = ObservationCandidateEvaluation::new(
            candidate("nisar", ObservationIntrusiveness::NonInvasive),
            1.5,
            vec![HypothesisInformationGain::new("cavity", 0.4).unwrap()],
            known_cost(1.0, 2.0, 0.01),
            model(),
            vec![],
        )
        .unwrap();
        assert_eq!(evaluation.expected_information_gain_nats, 1.5);
        // There is intentionally no execution token/capability in the type.
    }

    #[test]
    fn clearly_dominated_candidate_is_removed_from_frontier() {
        let better = ObservationCandidateEvaluation::new(
            candidate("better", ObservationIntrusiveness::NonInvasive),
            2.0,
            vec![],
            known_cost(1.0, 1.0, 0.01),
            model(),
            vec![],
        )
        .unwrap();
        let worse = ObservationCandidateEvaluation::new(
            candidate("worse", ObservationIntrusiveness::NonInvasive),
            1.0,
            vec![],
            ObservationCostVector {
                delay_seconds: Some(2.0),
                energy_joules: Some(2.0),
                downlink_bytes: Some(20_000.0),
                human_hours: Some(2.0),
                monetary_cost: Some((200.0, "USD".into())),
                operational_risk: Some(0.02),
            },
            model(),
            vec![],
        )
        .unwrap();

        let plan = ObservationPlan::new(vec![better, worse]).unwrap();
        let ids: Vec<&str> = plan
            .pareto_frontier()
            .into_iter()
            .map(|evaluation| evaluation.candidate.id.as_str())
            .collect();
        assert_eq!(ids, vec!["better"]);
    }

    #[test]
    fn unknown_cost_does_not_get_treated_as_free() {
        let known = ObservationCandidateEvaluation::new(
            candidate("known", ObservationIntrusiveness::NonInvasive),
            2.0,
            vec![],
            known_cost(1.0, 1.0, 0.01),
            model(),
            vec![],
        )
        .unwrap();
        let unknown = ObservationCandidateEvaluation::new(
            candidate("unknown", ObservationIntrusiveness::NonInvasive),
            1.0,
            vec![],
            ObservationCostVector::unknown(),
            model(),
            vec![],
        )
        .unwrap();
        let plan = ObservationPlan::new(vec![known, unknown]).unwrap();
        assert_eq!(plan.pareto_frontier().len(), 2);
    }

    #[test]
    fn invasive_measurement_is_not_silently_preferred_over_equal_noninvasive_measurement() {
        let noninvasive = ObservationCandidateEvaluation::new(
            candidate("radar", ObservationIntrusiveness::NonInvasive),
            1.0,
            vec![],
            known_cost(1.0, 1.0, 0.01),
            model(),
            vec![],
        )
        .unwrap();
        let invasive = ObservationCandidateEvaluation::new(
            candidate("borehole", ObservationIntrusiveness::Invasive),
            1.0,
            vec![],
            known_cost(1.0, 1.0, 0.01),
            model(),
            vec![],
        )
        .unwrap();
        let plan = ObservationPlan::new(vec![noninvasive, invasive]).unwrap();
        let ids: Vec<&str> = plan
            .pareto_frontier()
            .into_iter()
            .map(|evaluation| evaluation.candidate.id.as_str())
            .collect();
        assert_eq!(ids, vec!["radar"]);
    }
}
