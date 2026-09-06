// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical physical-effect vocabulary for Symthaea.
//!
//! This crate deliberately contains no planner, solver, HAL, device driver, or
//! execution capability. It describes desired physical transitions and
//! *unqualified* intervention proposals so higher layers can reason about them
//! without accidentally acquiring actuator authority.
//!
//! Architectural invariant:
//!
//! ```text
//! CanModel(effect) != CanPropose(effect) != CanExecute(effect)
//! ```

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Broad physical mechanism families.
///
/// A modality is intentionally not a device type. Specialist crates own the
/// numerical physics and hardware adapters for each domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalModality {
    Mechanical,
    Acoustic,
    Photonic,
    Thermal,
    Electric,
    Magnetic,
    Fluid,
    Plasma,
    Chemical,
    Coupled,
}

/// Modality-neutral physical state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    Observe,
    Characterize,
    Communicate,
    Translate,
    Rotate,
    Heat,
    Cool,
    Excite,
    Illuminate,
    Constrain,
    Separate,
    Join,
    DepositMaterial,
    RemoveMaterial,
    AlterFlow,
    Custom,
}

/// Expected recoverability of a requested state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reversibility {
    Reversible,
    Recoverable,
    Irreversible,
}

/// Maximum authority required before a proposal may eventually reach an
/// execution boundary.
///
/// The ordering is intentional. Higher variants imply strictly more authority.
/// This crate does not mint that authority.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityClass {
    #[default]
    SimulationOnly,
    PassiveObservation,
    DiagnosticExcitation,
    ReversibleActuation,
    ControlledEnergyTransfer,
    IrreversibleMaterialChange,
}

impl AuthorityClass {
    /// Whether this authority level is at least the requested level.
    pub fn allows(self, required: Self) -> bool {
        self >= required
    }
}

/// Stable reference to a region in an external world/geometry model.
///
/// Geometry remains owned by perception, CAD, digital-twin, or simulator
/// layers; this crate only carries an auditable reference.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetRegion {
    pub frame_id: String,
    pub region_id: String,
}

impl TargetRegion {
    pub fn new(frame_id: impl Into<String>, region_id: impl Into<String>) -> Self {
        Self {
            frame_id: frame_id.into(),
            region_id: region_id.into(),
        }
    }

    pub fn validate(&self) -> Result<(), EffectValidationError> {
        if self.frame_id.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("target.frame_id"));
        }
        if self.region_id.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("target.region_id"));
        }
        Ok(())
    }
}

/// Decision-relevant uncertainty limits for a desired transition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyBudget {
    /// Maximum model/knowledge uncertainty in [0, 1].
    pub max_epistemic: f64,
    /// Maximum irreducible/noise uncertainty in [0, 1].
    pub max_aleatoric: f64,
    /// Minimum calibrated success confidence in [0, 1].
    pub min_confidence: f64,
}

impl Default for UncertaintyBudget {
    fn default() -> Self {
        Self {
            max_epistemic: 1.0,
            max_aleatoric: 1.0,
            min_confidence: 0.0,
        }
    }
}

impl UncertaintyBudget {
    pub fn validate(&self) -> Result<(), EffectValidationError> {
        validate_probability("uncertainty.max_epistemic", self.max_epistemic)?;
        validate_probability("uncertainty.max_aleatoric", self.max_aleatoric)?;
        validate_probability("uncertainty.min_confidence", self.min_confidence)?;
        Ok(())
    }
}

/// Explicit resource/time envelope for a physical transition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ResourceBudget {
    /// Maximum net energy transfer in joules when known.
    pub max_energy_j: Option<f64>,
    /// Maximum instantaneous/average power in watts when known.
    pub max_power_w: Option<f64>,
    /// Maximum requested action duration in milliseconds when known.
    pub max_duration_ms: Option<u64>,
}

impl ResourceBudget {
    pub fn validate(&self) -> Result<(), EffectValidationError> {
        validate_optional_nonnegative("resources.max_energy_j", self.max_energy_j)?;
        validate_optional_nonnegative("resources.max_power_w", self.max_power_w)?;
        if self.max_duration_ms == Some(0) {
            return Err(EffectValidationError::InvalidBudget(
                "resources.max_duration_ms must be greater than zero when present".into(),
            ));
        }
        Ok(())
    }
}

/// A modality-neutral request for a physical state transition.
///
/// This is a *request to reason*, not an executable command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesiredTransition {
    pub id: String,
    pub objective: String,
    pub target: TargetRegion,
    pub effect: EffectKind,
    pub allowed_modalities: Vec<PhysicalModality>,
    pub required_authority: AuthorityClass,
    pub reversibility: Reversibility,
    pub uncertainty: UncertaintyBudget,
    pub resources: ResourceBudget,
}

impl DesiredTransition {
    /// Construct a transition that is explicitly restricted to simulation.
    pub fn simulation_only(
        id: impl Into<String>,
        objective: impl Into<String>,
        target: TargetRegion,
        effect: EffectKind,
        allowed_modalities: Vec<PhysicalModality>,
    ) -> Self {
        Self {
            id: id.into(),
            objective: objective.into(),
            target,
            effect,
            allowed_modalities,
            required_authority: AuthorityClass::SimulationOnly,
            reversibility: Reversibility::Reversible,
            uncertainty: UncertaintyBudget::default(),
            resources: ResourceBudget::default(),
        }
    }

    pub fn validate(&self) -> Result<(), EffectValidationError> {
        if self.id.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("transition.id"));
        }
        if self.objective.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("transition.objective"));
        }
        if self.allowed_modalities.is_empty() {
            return Err(EffectValidationError::NoAllowedModalities);
        }
        self.target.validate()?;
        self.uncertainty.validate()?;
        self.resources.validate()?;
        Ok(())
    }
}

/// Reference to a mechanism implementation supplied by another crate/adapter.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MechanismRef {
    pub backend: String,
    pub mechanism: String,
    pub modality: PhysicalModality,
}

impl MechanismRef {
    pub fn validate(&self) -> Result<(), EffectValidationError> {
        if self.backend.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("mechanism.backend"));
        }
        if self.mechanism.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("mechanism.name"));
        }
        Ok(())
    }
}

/// Distribution summary produced by a model/solver ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PredictedOutcome {
    pub success_probability: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
}

impl PredictedOutcome {
    pub fn validate(&self) -> Result<(), EffectValidationError> {
        validate_probability("outcome.success_probability", self.success_probability)?;
        validate_probability("outcome.epistemic_uncertainty", self.epistemic_uncertainty)?;
        validate_probability("outcome.aleatoric_uncertainty", self.aleatoric_uncertainty)?;
        Ok(())
    }
}

/// An unqualified mechanism-specific intervention candidate.
///
/// Deliberately contains no execution method, actuator handle, or authority token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposedIntervention {
    pub id: String,
    pub transition_id: String,
    pub mechanism: MechanismRef,
    pub required_authority: AuthorityClass,
    pub predicted_outcome: PredictedOutcome,
}

impl ProposedIntervention {
    pub fn validate(&self) -> Result<(), EffectValidationError> {
        if self.id.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("proposal.id"));
        }
        if self.transition_id.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("proposal.transition_id"));
        }
        self.mechanism.validate()?;
        self.predicted_outcome.validate()?;
        Ok(())
    }
}

/// A scenario Symthaea may model for prediction, resilience, or defense.
///
/// This type is intentionally distinct from [`ProposedIntervention`]. There is
/// no `From<ThreatScenario>` implementation and no execution-facing API here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreatScenario {
    pub id: String,
    pub description: String,
    pub modalities: Vec<PhysicalModality>,
    pub anticipated_effects: Vec<EffectKind>,
}

impl ThreatScenario {
    pub fn validate(&self) -> Result<(), EffectValidationError> {
        if self.id.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("threat.id"));
        }
        if self.description.trim().is_empty() {
            return Err(EffectValidationError::EmptyField("threat.description"));
        }
        if self.modalities.is_empty() {
            return Err(EffectValidationError::InvalidThreat(
                "at least one modality is required".into(),
            ));
        }
        if self.anticipated_effects.is_empty() {
            return Err(EffectValidationError::InvalidThreat(
                "at least one anticipated effect is required".into(),
            ));
        }
        Ok(())
    }
}

/// Normal planner outcomes include declining to intervene.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstentionReason {
    InsufficientEvidence,
    UncertaintyTooHigh,
    OutsideModelValidity,
    NoAllowedModality,
    SafetyReviewRequired,
    NoQualifiedAction,
    Other(String),
}

/// Pre-qualification planner disposition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlannerDisposition {
    Propose(ProposedIntervention),
    Abstain(AbstentionReason),
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EffectValidationError {
    #[error("required field is empty: {0}")]
    EmptyField(&'static str),
    #[error("no physical modalities are allowed")]
    NoAllowedModalities,
    #[error("invalid probability/uncertainty for {field}: {value}")]
    InvalidProbability { field: &'static str, value: f64 },
    #[error("invalid resource budget: {0}")]
    InvalidBudget(String),
    #[error("invalid threat scenario: {0}")]
    InvalidThreat(String),
}

fn validate_probability(field: &'static str, value: f64) -> Result<(), EffectValidationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EffectValidationError::InvalidProbability { field, value });
    }
    Ok(())
}

fn validate_optional_nonnegative(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), EffectValidationError> {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            return Err(EffectValidationError::InvalidBudget(format!(
                "{field} must be finite and non-negative, got {value}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> TargetRegion {
        TargetRegion::new("world", "fixture")
    }

    #[test]
    fn authority_order_is_monotonic() {
        assert!(AuthorityClass::SimulationOnly.allows(AuthorityClass::SimulationOnly));
        assert!(!AuthorityClass::PassiveObservation.allows(AuthorityClass::ReversibleActuation));
        assert!(
            AuthorityClass::ControlledEnergyTransfer
                .allows(AuthorityClass::DiagnosticExcitation)
        );
    }

    #[test]
    fn simulation_constructor_fails_closed() {
        let transition = DesiredTransition::simulation_only(
            "t-1",
            "characterize fixture",
            target(),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        assert_eq!(
            transition.required_authority,
            AuthorityClass::SimulationOnly
        );
        transition.validate().unwrap();
    }

    #[test]
    fn non_finite_uncertainty_is_rejected() {
        let mut transition = DesiredTransition::simulation_only(
            "t-2",
            "observe fixture",
            target(),
            EffectKind::Observe,
            vec![PhysicalModality::Photonic],
        );
        transition.uncertainty.max_epistemic = f64::NAN;
        assert!(matches!(
            transition.validate(),
            Err(EffectValidationError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn negative_resource_budget_is_rejected() {
        let mut transition = DesiredTransition::simulation_only(
            "t-3",
            "simulate thermal transition",
            target(),
            EffectKind::Heat,
            vec![PhysicalModality::Thermal],
        );
        transition.resources.max_energy_j = Some(-1.0);
        assert!(matches!(
            transition.validate(),
            Err(EffectValidationError::InvalidBudget(_))
        ));
    }

    #[test]
    fn empty_modality_set_is_not_implicitly_anything() {
        let transition = DesiredTransition::simulation_only(
            "t-4",
            "ambiguous transition",
            target(),
            EffectKind::Custom,
            vec![],
        );
        assert_eq!(
            transition.validate(),
            Err(EffectValidationError::NoAllowedModalities)
        );
    }

    #[test]
    fn proposal_is_validated_but_never_qualified_here() {
        let proposal = ProposedIntervention {
            id: "p-1".into(),
            transition_id: "t-1".into(),
            mechanism: MechanismRef {
                backend: "reference".into(),
                mechanism: "simulated_probe".into(),
                modality: PhysicalModality::Acoustic,
            },
            required_authority: AuthorityClass::SimulationOnly,
            predicted_outcome: PredictedOutcome {
                success_probability: 0.9,
                epistemic_uncertainty: 0.1,
                aleatoric_uncertainty: 0.05,
            },
        };
        proposal.validate().unwrap();
    }

    #[test]
    fn threat_model_is_a_separate_nonempty_schema() {
        let threat = ThreatScenario {
            id: "threat-1".into(),
            description: "environmental energetic disturbance".into(),
            modalities: vec![PhysicalModality::Coupled],
            anticipated_effects: vec![EffectKind::Heat, EffectKind::Excite],
        };
        threat.validate().unwrap();
    }
}
