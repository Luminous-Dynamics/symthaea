//! Evidence-bearing counterfactual scenario envelopes for Symthaea.
//!
//! A scenario is not a prediction, an identified causal effect is not a
//! simulation result, and a simulation result is not authority to act. This
//! crate gives those distinctions durable types before Planetary Perception,
//! digital twins, Symtropy, or Mycelix begin exchanging counterfactuals.

use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_causal_reasoning::counterfactual::{
    CausalAssumption, IdentificationMethod, UnidentifiedReason,
};
use symthaea_digital_twin::Intervention;
use symthaea_earth_causal_query::EarthCausalQueryOutcome;

pub type Result<T> = std::result::Result<T, ScenarioError>;

#[derive(Debug, Clone, PartialEq)]
pub enum ScenarioError {
    EmptyField(&'static str),
    MissingBaselineEvidence,
    MissingModel,
    MissingSpeculativeAssumption,
    InvalidUrgency(f64),
    NonFiniteScore { field: &'static str, value: f64 },
}

impl Display for ScenarioError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::MissingBaselineEvidence => {
                write!(f, "a scenario requires at least one baseline evidence reference")
            }
            Self::MissingModel => write!(f, "a simulated scenario requires at least one model"),
            Self::MissingSpeculativeAssumption => {
                write!(f, "a speculative scenario must state at least one assumption")
            }
            Self::InvalidUrgency(value) => {
                write!(f, "scenario urgency must be finite and in [0, 1], got {value}")
            }
            Self::NonFiniteScore { field, value } => {
                write!(f, "{field} must be finite, got {value}")
            }
        }
    }
}

impl Error for ScenarioError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(ScenarioError::EmptyField(field));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioModelRef {
    pub model_id: String,
    pub version: String,
    pub role: ScenarioModelRole,
    pub artifact_digest: Option<String>,
}

impl ScenarioModelRef {
    pub fn new(
        model_id: impl Into<String>,
        version: impl Into<String>,
        role: ScenarioModelRole,
    ) -> Result<Self> {
        let model_id = model_id.into();
        let version = version.into();
        non_empty(&model_id, "scenario model id")?;
        non_empty(&version, "scenario model version")?;
        Ok(Self {
            model_id,
            version,
            role,
            artifact_digest: None,
        })
    }

    pub fn with_artifact_digest(mut self, digest: impl Into<String>) -> Result<Self> {
        let digest = digest.into();
        non_empty(&digest, "scenario model artifact digest")?;
        self.artifact_digest = Some(digest);
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioModelRole {
    Causal,
    DigitalTwin,
    PhysicsSimulation,
    Symtropy,
    Statistical,
    Learned,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioIntervention {
    pub target: String,
    pub action: String,
    pub rationale: String,
    /// Context for prioritization only. This is not execution authority.
    pub urgency: Option<f64>,
}

impl ScenarioIntervention {
    pub fn new(
        target: impl Into<String>,
        action: impl Into<String>,
        rationale: impl Into<String>,
    ) -> Result<Self> {
        let target = target.into();
        let action = action.into();
        let rationale = rationale.into();
        non_empty(&target, "intervention target")?;
        non_empty(&action, "intervention action")?;
        non_empty(&rationale, "intervention rationale")?;
        Ok(Self {
            target,
            action,
            rationale,
            urgency: None,
        })
    }

    pub fn with_urgency(mut self, urgency: f64) -> Result<Self> {
        if !urgency.is_finite() || !(0.0..=1.0).contains(&urgency) {
            return Err(ScenarioError::InvalidUrgency(urgency));
        }
        self.urgency = Some(urgency);
        Ok(self)
    }
}

impl TryFrom<&Intervention> for ScenarioIntervention {
    type Error = ScenarioError;

    fn try_from(value: &Intervention) -> Result<Self> {
        Self::new(&value.twin_id, &value.action, &value.rationale)?.with_urgency(value.urgency)
    }
}

/// Epistemic status of the causal component of a scenario.
#[derive(Debug, Clone)]
pub enum CausalSupport {
    NotEvaluated,
    Identified {
        estimand_description: String,
        method: IdentificationMethod,
        identification_confidence: f64,
        adjustment_evidence_ids: Vec<String>,
    },
    Unidentified {
        reason: UnidentifiedReason,
        missing: Vec<String>,
        suggestions: Vec<String>,
    },
    AssumptionRequired {
        assumption: CausalAssumption,
        estimand_description: String,
        adjustment_evidence_ids: Vec<String>,
        plausibility: f64,
    },
}

impl CausalSupport {
    pub fn from_query_outcome(outcome: EarthCausalQueryOutcome) -> Result<Self> {
        match outcome {
            EarthCausalQueryOutcome::Identified {
                estimand_description,
                method,
                identification_confidence,
                adjustment_evidence_ids,
            } => {
                if !identification_confidence.is_finite() {
                    return Err(ScenarioError::NonFiniteScore {
                        field: "identification confidence",
                        value: identification_confidence,
                    });
                }
                Ok(Self::Identified {
                    estimand_description,
                    method,
                    identification_confidence,
                    adjustment_evidence_ids,
                })
            }
            EarthCausalQueryOutcome::Unidentified {
                reason,
                missing,
                suggestions,
            } => Ok(Self::Unidentified {
                reason,
                missing,
                suggestions,
            }),
            EarthCausalQueryOutcome::AssumptionRequired {
                assumption,
                estimand_description,
                adjustment_evidence_ids,
                plausibility,
            } => {
                if !plausibility.is_finite() {
                    return Err(ScenarioError::NonFiniteScore {
                        field: "assumption plausibility",
                        value: plausibility,
                    });
                }
                Ok(Self::AssumptionRequired {
                    assumption,
                    estimand_description,
                    adjustment_evidence_ids,
                    plausibility,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioEpistemicClass {
    CausallyIdentified,
    CausallyUnidentified,
    AssumptionDependent,
    SimulationOnly,
    Speculative,
}

/// Provenance envelope for one counterfactual scenario.
///
/// There is intentionally no authority, execution token, or aggregate utility
/// score in this type. It describes a possible world and the evidence/model
/// basis used to construct it; it does not authorize changing the real world.
#[derive(Debug, Clone)]
pub struct CounterfactualScenarioEnvelope {
    id: String,
    baseline_evidence_ids: Vec<String>,
    intervention: ScenarioIntervention,
    models: Vec<ScenarioModelRef>,
    causal_support: CausalSupport,
    epistemic_class: ScenarioEpistemicClass,
    assumptions: Vec<String>,
}

impl CounterfactualScenarioEnvelope {
    pub fn from_causal_query(
        id: impl Into<String>,
        baseline_evidence_ids: Vec<String>,
        intervention: ScenarioIntervention,
        models: Vec<ScenarioModelRef>,
        outcome: EarthCausalQueryOutcome,
        assumptions: Vec<String>,
    ) -> Result<Self> {
        let causal_support = CausalSupport::from_query_outcome(outcome)?;
        let epistemic_class = match &causal_support {
            CausalSupport::Identified { .. } => ScenarioEpistemicClass::CausallyIdentified,
            CausalSupport::Unidentified { .. } => ScenarioEpistemicClass::CausallyUnidentified,
            CausalSupport::AssumptionRequired { .. } => {
                ScenarioEpistemicClass::AssumptionDependent
            }
            CausalSupport::NotEvaluated => unreachable!("query outcome is always evaluated"),
        };
        Self::build(
            id,
            baseline_evidence_ids,
            intervention,
            models,
            causal_support,
            epistemic_class,
            assumptions,
        )
    }

    pub fn simulation_only(
        id: impl Into<String>,
        baseline_evidence_ids: Vec<String>,
        intervention: ScenarioIntervention,
        models: Vec<ScenarioModelRef>,
        assumptions: Vec<String>,
    ) -> Result<Self> {
        if models.is_empty() {
            return Err(ScenarioError::MissingModel);
        }
        Self::build(
            id,
            baseline_evidence_ids,
            intervention,
            models,
            CausalSupport::NotEvaluated,
            ScenarioEpistemicClass::SimulationOnly,
            assumptions,
        )
    }

    pub fn speculative(
        id: impl Into<String>,
        baseline_evidence_ids: Vec<String>,
        intervention: ScenarioIntervention,
        models: Vec<ScenarioModelRef>,
        assumptions: Vec<String>,
    ) -> Result<Self> {
        if assumptions.is_empty() {
            return Err(ScenarioError::MissingSpeculativeAssumption);
        }
        Self::build(
            id,
            baseline_evidence_ids,
            intervention,
            models,
            CausalSupport::NotEvaluated,
            ScenarioEpistemicClass::Speculative,
            assumptions,
        )
    }

    fn build(
        id: impl Into<String>,
        baseline_evidence_ids: Vec<String>,
        intervention: ScenarioIntervention,
        models: Vec<ScenarioModelRef>,
        causal_support: CausalSupport,
        epistemic_class: ScenarioEpistemicClass,
        assumptions: Vec<String>,
    ) -> Result<Self> {
        let id = id.into();
        non_empty(&id, "scenario id")?;
        if baseline_evidence_ids.is_empty() {
            return Err(ScenarioError::MissingBaselineEvidence);
        }
        for evidence_id in &baseline_evidence_ids {
            non_empty(evidence_id, "baseline evidence id")?;
        }
        for assumption in &assumptions {
            non_empty(assumption, "scenario assumption")?;
        }
        for model in &models {
            non_empty(&model.model_id, "scenario model id")?;
            non_empty(&model.version, "scenario model version")?;
            if let Some(digest) = &model.artifact_digest {
                non_empty(digest, "scenario model artifact digest")?;
            }
        }

        Ok(Self {
            id,
            baseline_evidence_ids,
            intervention,
            models,
            causal_support,
            epistemic_class,
            assumptions,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn baseline_evidence_ids(&self) -> &[String] {
        &self.baseline_evidence_ids
    }

    pub fn intervention(&self) -> &ScenarioIntervention {
        &self.intervention
    }

    pub fn models(&self) -> &[ScenarioModelRef] {
        &self.models
    }

    pub fn causal_support(&self) -> &CausalSupport {
        &self.causal_support
    }

    pub const fn epistemic_class(&self) -> ScenarioEpistemicClass {
        self.epistemic_class
    }

    pub fn assumptions(&self) -> &[String] {
        &self.assumptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_causal_reasoning::counterfactual::{CausalEstimand, CausalQueryOutcome};
    use symthaea_earth_causal_query::EarthCausalQueryOutcome;

    fn intervention() -> ScenarioIntervention {
        ScenarioIntervention::new("wetland-1", "restore", "compare restoration scenario").unwrap()
    }

    fn model() -> ScenarioModelRef {
        ScenarioModelRef::new("watershed-twin", "0.1.0", ScenarioModelRole::DigitalTwin)
            .unwrap()
            .with_artifact_digest("sha256:fixture-model")
            .unwrap()
    }

    #[test]
    fn identified_query_produces_identified_scenario_class() {
        let outcome = EarthCausalQueryOutcome::Identified {
            estimand_description: "P(water_extent|do(restoration))".into(),
            method: IdentificationMethod::BackdoorAdjustment,
            identification_confidence: 0.9,
            adjustment_evidence_ids: vec!["rainfall".into()],
        };

        let scenario = CounterfactualScenarioEnvelope::from_causal_query(
            "scenario-1",
            vec!["baseline-obs".into()],
            intervention(),
            vec![model()],
            outcome,
            vec!["stable measurement process".into()],
        )
        .unwrap();

        assert_eq!(
            scenario.epistemic_class(),
            ScenarioEpistemicClass::CausallyIdentified
        );
    }

    #[test]
    fn unidentified_query_stays_unidentified() {
        let outcome = EarthCausalQueryOutcome::Unidentified {
            reason: UnidentifiedReason::NotConnected,
            missing: vec![],
            suggestions: vec![],
        };

        let scenario = CounterfactualScenarioEnvelope::from_causal_query(
            "scenario-2",
            vec!["baseline-obs".into()],
            intervention(),
            vec![model()],
            outcome,
            vec![],
        )
        .unwrap();

        assert_eq!(
            scenario.epistemic_class(),
            ScenarioEpistemicClass::CausallyUnidentified
        );
    }

    #[test]
    fn simulation_only_requires_a_model() {
        assert_eq!(
            CounterfactualScenarioEnvelope::simulation_only(
                "scenario-3",
                vec!["baseline-obs".into()],
                intervention(),
                vec![],
                vec![],
            )
            .unwrap_err(),
            ScenarioError::MissingModel
        );
    }

    #[test]
    fn speculative_scenario_requires_explicit_assumption() {
        assert_eq!(
            CounterfactualScenarioEnvelope::speculative(
                "scenario-4",
                vec!["baseline-obs".into()],
                intervention(),
                vec![model()],
                vec![],
            )
            .unwrap_err(),
            ScenarioError::MissingSpeculativeAssumption
        );
    }

    #[test]
    fn digital_twin_intervention_maps_without_execution_authority() {
        let twin = Intervention {
            twin_id: "pump-7".into(),
            action: "inspect".into(),
            rationale: "rising residual".into(),
            urgency: 0.8,
        };
        let mapped = ScenarioIntervention::try_from(&twin).unwrap();
        assert_eq!(mapped.target, "pump-7");
        assert_eq!(mapped.action, "inspect");
        assert_eq!(mapped.urgency, Some(0.8));
    }

    #[test]
    fn invalid_urgency_is_rejected() {
        assert_eq!(
            intervention().with_urgency(1.5).unwrap_err(),
            ScenarioError::InvalidUrgency(1.5)
        );
    }

    #[test]
    fn raw_causal_estimand_placeholder_is_not_part_of_scenario_api() {
        // Document the upstream semantic hazard in an executable test fixture:
        // symbolic identification may carry an effect placeholder of zero.
        let upstream = CausalQueryOutcome::Identified {
            estimand: CausalEstimand {
                effect: 0.0,
                adjustment_set: vec![],
                description: "symbolic estimand only".into(),
            },
            method: IdentificationMethod::BackdoorAdjustment,
            confidence: 0.9,
        };
        match upstream {
            CausalQueryOutcome::Identified { estimand, .. } => {
                assert_eq!(estimand.effect, 0.0);
            }
            _ => unreachable!(),
        }
        // CounterfactualScenarioEnvelope has no numerical effect field by design.
    }
}
