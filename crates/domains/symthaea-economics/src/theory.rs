// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Falsifiable economic claims, hard constraints, and explicit normative
//! propositions.
//!
//! Economic schools are intentionally not first-class authority objects here.
//! The kernel represents decomposed claims that different schools or model
//! families may share, contradict, or implement differently.

use std::collections::BTreeSet;

use crate::error::{EconomicsError, Result};
use crate::ontology::{ClaimId, MechanismId, PredictionId, VariableId};
use crate::science::StatementKind;

fn require_text(value: impl Into<String>, context: &'static str) -> Result<String> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(EconomicsError::InvalidParameter { context })
    } else {
        Ok(value)
    }
}

fn ensure_unique<T: Ord>(values: &[T], context: &'static str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(EconomicsError::InvalidParameter { context });
        }
    }
    Ok(())
}

/// Direction of a falsifiable predicted response. `NonMonotonic` is explicit
/// rather than forcing a complex response into an increase/decrease binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseDirection {
    Increase,
    Decrease,
    ApproximatelyUnchanged,
    NonMonotonic,
}

/// Whether an empirical claim is only associational or asserts a mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmpiricalClaimMode {
    Associational,
    Mechanistic,
}

/// One prospective consequence that can be scored against observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prediction {
    id: PredictionId,
    outcome: VariableId,
    direction: ResponseDirection,
    horizon: String,
    condition: Option<String>,
}

impl Prediction {
    pub fn new(
        id: PredictionId,
        outcome: VariableId,
        direction: ResponseDirection,
        horizon: impl Into<String>,
        condition: Option<String>,
    ) -> Result<Self> {
        let horizon = require_text(horizon, "prediction horizon")?;
        if condition.as_ref().is_some_and(|value| value.trim().is_empty()) {
            return Err(EconomicsError::InvalidParameter {
                context: "prediction condition",
            });
        }
        Ok(Self {
            id,
            outcome,
            direction,
            horizon,
            condition,
        })
    }

    pub fn id(&self) -> &PredictionId {
        &self.id
    }

    pub fn outcome(&self) -> &VariableId {
        &self.outcome
    }

    pub fn direction(&self) -> ResponseDirection {
        self.direction
    }

    pub fn horizon(&self) -> &str {
        &self.horizon
    }

    pub fn condition(&self) -> Option<&str> {
        self.condition.as_deref()
    }
}

/// A predeclared observation that would count against one specific prediction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FalsificationCriterion {
    prediction: PredictionId,
    description: String,
}

impl FalsificationCriterion {
    pub fn new(prediction: PredictionId, description: impl Into<String>) -> Result<Self> {
        Ok(Self {
            prediction,
            description: require_text(description, "falsification criterion")?,
        })
    }

    pub fn prediction(&self) -> &PredictionId {
        &self.prediction
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

/// An accounting, conservation, or other declared hard constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintClaim {
    id: ClaimId,
    statement: String,
    invariant: String,
    variables: Vec<VariableId>,
}

impl ConstraintClaim {
    pub fn new(
        id: ClaimId,
        statement: impl Into<String>,
        invariant: impl Into<String>,
        variables: Vec<VariableId>,
    ) -> Result<Self> {
        if variables.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "constraint variables",
            });
        }
        ensure_unique(&variables, "duplicate constraint variable")?;
        Ok(Self {
            id,
            statement: require_text(statement, "constraint statement")?,
            invariant: require_text(invariant, "constraint invariant")?,
            variables,
        })
    }

    pub fn id(&self) -> &ClaimId {
        &self.id
    }

    pub fn statement(&self) -> &str {
        &self.statement
    }

    pub fn invariant(&self) -> &str {
        &self.invariant
    }

    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

/// A falsifiable proposition about the observed economy. Evidence status is
/// intentionally not a caller-provided field: later evidence qualification
/// must assess a claim externally rather than letting the claim self-certify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmpiricalClaim {
    id: ClaimId,
    statement: String,
    mode: EmpiricalClaimMode,
    scope: String,
    mechanisms: Vec<MechanismId>,
    predictions: Vec<Prediction>,
    falsifiers: Vec<FalsificationCriterion>,
}

impl EmpiricalClaim {
    pub fn new(
        id: ClaimId,
        statement: impl Into<String>,
        mode: EmpiricalClaimMode,
        scope: impl Into<String>,
        mechanisms: Vec<MechanismId>,
        predictions: Vec<Prediction>,
        falsifiers: Vec<FalsificationCriterion>,
    ) -> Result<Self> {
        if predictions.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "empirical predictions",
            });
        }
        if falsifiers.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "empirical falsifiers",
            });
        }
        ensure_unique(&mechanisms, "duplicate empirical mechanism")?;
        if mode == EmpiricalClaimMode::Mechanistic && mechanisms.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "mechanistic claim mechanisms",
            });
        }
        if mode == EmpiricalClaimMode::Associational && !mechanisms.is_empty() {
            return Err(EconomicsError::InvalidParameter {
                context: "associational claim cannot assert mechanisms",
            });
        }

        let prediction_ids: Vec<&PredictionId> = predictions.iter().map(Prediction::id).collect();
        ensure_unique(&prediction_ids, "duplicate empirical prediction id")?;
        for falsifier in &falsifiers {
            if !prediction_ids.contains(&falsifier.prediction()) {
                return Err(EconomicsError::InvalidParameter {
                    context: "falsifier references unknown prediction",
                });
            }
        }
        for prediction_id in &prediction_ids {
            if !falsifiers
                .iter()
                .any(|falsifier| falsifier.prediction() == *prediction_id)
            {
                return Err(EconomicsError::InvalidParameter {
                    context: "prediction lacks a falsification criterion",
                });
            }
        }

        Ok(Self {
            id,
            statement: require_text(statement, "empirical statement")?,
            mode,
            scope: require_text(scope, "empirical scope")?,
            mechanisms,
            predictions,
            falsifiers,
        })
    }

    pub fn id(&self) -> &ClaimId {
        &self.id
    }

    pub fn statement(&self) -> &str {
        &self.statement
    }

    pub fn mode(&self) -> EmpiricalClaimMode {
        self.mode
    }

    pub fn scope(&self) -> &str {
        &self.scope
    }

    pub fn mechanisms(&self) -> &[MechanismId] {
        &self.mechanisms
    }

    pub fn predictions(&self) -> &[Prediction] {
        &self.predictions
    }

    pub fn falsifiers(&self) -> &[FalsificationCriterion] {
        &self.falsifiers
    }
}

/// A value judgment supplied explicitly by people or governance. It is not an
/// empirical claim and cannot be promoted from empirical evidence by this API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormativeProposition {
    id: ClaimId,
    statement: String,
    value_dimensions: Vec<String>,
}

impl NormativeProposition {
    pub fn new(
        id: ClaimId,
        statement: impl Into<String>,
        value_dimensions: Vec<String>,
    ) -> Result<Self> {
        if value_dimensions.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "normative value dimensions",
            });
        }
        for dimension in &value_dimensions {
            if dimension.trim().is_empty() {
                return Err(EconomicsError::InvalidParameter {
                    context: "normative value dimension",
                });
            }
        }
        ensure_unique(&value_dimensions, "duplicate normative value dimension")?;
        Ok(Self {
            id,
            statement: require_text(statement, "normative statement")?,
            value_dimensions,
        })
    }

    pub fn id(&self) -> &ClaimId {
        &self.id
    }

    pub fn statement(&self) -> &str {
        &self.statement
    }

    pub fn value_dimensions(&self) -> &[String] {
        &self.value_dimensions
    }
}

/// A closed sum type that prevents constraint, empirical, and normative
/// statements from being silently treated as interchangeable records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EconomicClaim {
    Constraint(ConstraintClaim),
    Empirical(EmpiricalClaim),
    Normative(NormativeProposition),
}

impl EconomicClaim {
    pub fn id(&self) -> &ClaimId {
        match self {
            Self::Constraint(claim) => claim.id(),
            Self::Empirical(claim) => claim.id(),
            Self::Normative(claim) => claim.id(),
        }
    }

    pub fn kind(&self) -> StatementKind {
        match self {
            Self::Constraint(_) => StatementKind::Constraint,
            Self::Empirical(_) => StatementKind::Empirical,
            Self::Normative(_) => StatementKind::Normative,
        }
    }

    pub fn referenced_variables(&self) -> Vec<&VariableId> {
        match self {
            Self::Constraint(claim) => claim.variables().iter().collect(),
            Self::Empirical(claim) => claim
                .predictions()
                .iter()
                .map(Prediction::outcome)
                .collect(),
            Self::Normative(_) => Vec::new(),
        }
    }

    pub fn referenced_mechanisms(&self) -> &[MechanismId] {
        match self {
            Self::Empirical(claim) => claim.mechanisms(),
            Self::Constraint(_) | Self::Normative(_) => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variable(id: &str) -> VariableId {
        VariableId::new(id).unwrap()
    }

    #[test]
    fn mechanistic_claim_requires_mechanism_prediction_and_falsifier() {
        let prediction = Prediction::new(
            PredictionId::new("p1").unwrap(),
            variable("employment"),
            ResponseDirection::Decrease,
            "12_months",
            None,
        )
        .unwrap();
        let claim = EmpiricalClaim::new(
            ClaimId::new("sticky_wage_demand_shock").unwrap(),
            "Negative nominal-demand shocks reduce employment when adjustment is slow.",
            EmpiricalClaimMode::Mechanistic,
            "unused_capacity_and_slow_nominal_adjustment",
            vec![MechanismId::new("nominal_rigidity").unwrap()],
            vec![prediction],
            vec![FalsificationCriterion::new(
                PredictionId::new("p1").unwrap(),
                "Comparable shocks show no systematic employment response.",
            )
            .unwrap()],
        )
        .unwrap();
        assert_eq!(claim.mode(), EmpiricalClaimMode::Mechanistic);
        assert_eq!(claim.predictions().len(), 1);
    }

    #[test]
    fn associational_claim_cannot_smuggle_causal_mechanism() {
        let prediction = Prediction::new(
            PredictionId::new("p1").unwrap(),
            variable("defaults"),
            ResponseDirection::Increase,
            "1_year",
            None,
        )
        .unwrap();
        let result = EmpiricalClaim::new(
            ClaimId::new("leverage_defaults_assoc").unwrap(),
            "Higher leverage is associated with more defaults.",
            EmpiricalClaimMode::Associational,
            "sample_scope",
            vec![MechanismId::new("balance_sheet_amplification").unwrap()],
            vec![prediction],
            vec![FalsificationCriterion::new(
                PredictionId::new("p1").unwrap(),
                "Out-of-sample association is absent or reversed.",
            )
            .unwrap()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn falsifier_must_target_declared_prediction() {
        let result = EmpiricalClaim::new(
            ClaimId::new("claim").unwrap(),
            "Statement",
            EmpiricalClaimMode::Associational,
            "scope",
            vec![],
            vec![Prediction::new(
                PredictionId::new("declared").unwrap(),
                variable("x"),
                ResponseDirection::Increase,
                "1_step",
                None,
            )
            .unwrap()],
            vec![FalsificationCriterion::new(
                PredictionId::new("other").unwrap(),
                "Wrong target",
            )
            .unwrap()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn every_prediction_requires_a_falsifier() {
        let result = EmpiricalClaim::new(
            ClaimId::new("partially_falsifiable").unwrap(),
            "Two outcomes are predicted, but only one has a falsifier.",
            EmpiricalClaimMode::Associational,
            "scope",
            vec![],
            vec![
                Prediction::new(
                    PredictionId::new("p1").unwrap(),
                    variable("x"),
                    ResponseDirection::Increase,
                    "1_step",
                    None,
                )
                .unwrap(),
                Prediction::new(
                    PredictionId::new("p2").unwrap(),
                    variable("y"),
                    ResponseDirection::Decrease,
                    "1_step",
                    None,
                )
                .unwrap(),
            ],
            vec![FalsificationCriterion::new(
                PredictionId::new("p1").unwrap(),
                "p1 fails out of sample.",
            )
            .unwrap()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn normative_proposition_remains_a_distinct_kind() {
        let claim = EconomicClaim::Normative(
            NormativeProposition::new(
                ClaimId::new("protect_agency").unwrap(),
                "Economic institutions should preserve human agency.",
                vec!["agency".into(), "freedom".into()],
            )
            .unwrap(),
        );
        assert_eq!(claim.kind(), StatementKind::Normative);
        assert!(claim.referenced_mechanisms().is_empty());
    }
}
