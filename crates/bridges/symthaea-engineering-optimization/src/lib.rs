// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Algorithm-neutral engineering design-space and Pareto evidence contracts.
//!
//! This crate deliberately does **not** implement an optimizer. Existing or future
//! NSGA-II, Bayesian, evolutionary, gradient, surrogate, or exhaustive-search
//! backends may consume these types, but they may not erase the distinction
//! between feasibility, objective values, uncertainty, or evaluation fidelity.
//!
//! In particular:
//!
//! `SurrogatePrediction != NumericalExecution != PhysicalMeasurement`
//!
//! and a Pareto frontier is descriptive evidence, not automatic product-selection
//! authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_sim_bridge::UncertaintyEstimate;
use thiserror::Error;

/// A variable domain admitted by an engineering design-space profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariableDomain {
    Continuous { lower: f64, upper: f64 },
    Discrete { allowed: Vec<i64> },
    Categorical { allowed: Vec<String> },
}

/// One named design variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignVariable {
    pub id: String,
    pub unit: Option<String>,
    pub domain: VariableDomain,
}

/// Exact value assigned to one design variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DesignValue {
    Continuous(f64),
    Discrete(i64),
    Categorical(String),
}

/// One exact candidate assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignAssignment {
    pub variable_id: String,
    pub value: DesignValue,
}

/// Candidate identity plus the exact parameter set evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignCandidate {
    pub id: String,
    /// Must be strictly ordered by `variable_id` and cover the complete profile.
    pub assignments: Vec<DesignAssignment>,
}

/// Optimization direction for a named physical/engineering objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

/// One objective in a profile. There is intentionally no objective weight here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    pub id: String,
    pub unit: String,
    pub direction: ObjectiveDirection,
}

/// Hard-constraint admission rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintRule {
    AtMost(f64),
    AtLeast(f64),
    BetweenInclusive { lower: f64, upper: f64 },
}

/// One hard feasibility constraint evaluated against a named metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardConstraintSpec {
    pub id: String,
    pub metric_id: String,
    pub unit: String,
    pub rule: ConstraintRule,
}

/// Complete objective/constraint/variable definition for one design campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationProfile {
    pub id: String,
    /// Strictly ordered by variable ID.
    pub variables: Vec<DesignVariable>,
    /// Strictly ordered by objective ID.
    pub objectives: Vec<ObjectiveSpec>,
    /// Strictly ordered by constraint ID.
    pub hard_constraints: Vec<HardConstraintSpec>,
}

impl OptimizationProfile {
    /// Validate canonical ordering, bounds, units, and IDs.
    pub fn validate(&self) -> Result<(), OptimizationError> {
        require_id("profile.id", &self.id)?;
        if self.variables.is_empty() {
            return Err(OptimizationError::EmptyCollection("profile.variables"));
        }
        if self.objectives.is_empty() {
            return Err(OptimizationError::EmptyCollection("profile.objectives"));
        }

        ensure_strictly_ordered(
            "profile.variables",
            self.variables.iter().map(|value| value.id.as_str()),
        )?;
        ensure_strictly_ordered(
            "profile.objectives",
            self.objectives.iter().map(|value| value.id.as_str()),
        )?;
        ensure_strictly_ordered(
            "profile.hard_constraints",
            self.hard_constraints.iter().map(|value| value.id.as_str()),
        )?;

        for variable in &self.variables {
            require_id("variable.id", &variable.id)?;
            if variable.unit.as_ref().is_some_and(|unit| unit.trim().is_empty()) {
                return Err(OptimizationError::EmptyIdentifier("variable.unit"));
            }
            validate_domain(&variable.domain)?;
        }
        for objective in &self.objectives {
            require_id("objective.id", &objective.id)?;
            require_id("objective.unit", &objective.unit)?;
        }
        for constraint in &self.hard_constraints {
            require_id("constraint.id", &constraint.id)?;
            require_id("constraint.metric_id", &constraint.metric_id)?;
            require_id("constraint.unit", &constraint.unit)?;
            validate_constraint_rule(&constraint.rule)?;
        }
        Ok(())
    }

    /// Stable digest over the complete canonical variable/objective/constraint profile.
    ///
    /// This is an identity aid, not a cryptographic proof that a solver behaved
    /// correctly. Canonical ordering is required before the digest is admitted.
    pub fn digest(&self) -> Result<String, OptimizationError> {
        self.validate()?;
        let mut material = String::new();
        push_field(&mut material, "profile", &self.id);
        for variable in &self.variables {
            push_field(&mut material, "var", &variable.id);
            push_field(&mut material, "unit", variable.unit.as_deref().unwrap_or(""));
            match &variable.domain {
                VariableDomain::Continuous { lower, upper } => {
                    push_field(&mut material, "kind", "continuous");
                    push_f64(&mut material, "lower", *lower);
                    push_f64(&mut material, "upper", *upper);
                }
                VariableDomain::Discrete { allowed } => {
                    push_field(&mut material, "kind", "discrete");
                    for value in allowed {
                        push_field(&mut material, "allowed_i64", &value.to_string());
                    }
                }
                VariableDomain::Categorical { allowed } => {
                    push_field(&mut material, "kind", "categorical");
                    for value in allowed {
                        push_field(&mut material, "allowed_category", value);
                    }
                }
            }
        }
        for objective in &self.objectives {
            push_field(&mut material, "objective", &objective.id);
            push_field(&mut material, "unit", &objective.unit);
            push_field(
                &mut material,
                "direction",
                match objective.direction {
                    ObjectiveDirection::Minimize => "min",
                    ObjectiveDirection::Maximize => "max",
                },
            );
        }
        for constraint in &self.hard_constraints {
            push_field(&mut material, "constraint", &constraint.id);
            push_field(&mut material, "metric", &constraint.metric_id);
            push_field(&mut material, "unit", &constraint.unit);
            match constraint.rule {
                ConstraintRule::AtMost(maximum) => {
                    push_field(&mut material, "rule", "at-most");
                    push_f64(&mut material, "maximum", maximum);
                }
                ConstraintRule::AtLeast(minimum) => {
                    push_field(&mut material, "rule", "at-least");
                    push_f64(&mut material, "minimum", minimum);
                }
                ConstraintRule::BetweenInclusive { lower, upper } => {
                    push_field(&mut material, "rule", "between-inclusive");
                    push_f64(&mut material, "lower", lower);
                    push_f64(&mut material, "upper", upper);
                }
            }
        }
        Ok(blake3::hash(material.as_bytes()).to_hex().to_string())
    }
}

/// Evidence fidelity/authority class for one candidate evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationFidelity {
    Analytical { model_id: String },
    Surrogate { model_id: String },
    Numerical { solver_id: String },
    Measured { protocol_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FidelityClass {
    Analytical,
    Surrogate,
    Numerical,
    Measured,
}

impl EvaluationFidelity {
    pub fn class(&self) -> FidelityClass {
        match self {
            Self::Analytical { .. } => FidelityClass::Analytical,
            Self::Surrogate { .. } => FidelityClass::Surrogate,
            Self::Numerical { .. } => FidelityClass::Numerical,
            Self::Measured { .. } => FidelityClass::Measured,
        }
    }

    fn validate(&self) -> Result<(), OptimizationError> {
        let (field, value) = match self {
            Self::Analytical { model_id } => ("fidelity.model_id", model_id),
            Self::Surrogate { model_id } => ("fidelity.model_id", model_id),
            Self::Numerical { solver_id } => ("fidelity.solver_id", solver_id),
            Self::Measured { protocol_id } => ("fidelity.protocol_id", protocol_id),
        };
        require_id(field, value)
    }
}

/// One objective metric observation/prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveObservation {
    pub objective_id: String,
    pub value: f64,
    pub unit: String,
    pub uncertainty: Option<UncertaintyEstimate>,
}

/// One hard-constraint metric evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintObservation {
    pub constraint_id: String,
    pub value: f64,
    pub unit: String,
    pub satisfied: bool,
}

/// Outcome state for one exact candidate evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationStatus {
    /// All hard constraints and all objectives were evaluated successfully.
    Complete,
    /// At least one hard constraint failed under this fidelity/source.
    Infeasible,
    /// The evaluator failed; no favorable defaults may be inferred.
    Failed { reason: String },
}

/// Evidence-bearing evaluation of one exact candidate under one exact profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub candidate: DesignCandidate,
    pub profile_digest: String,
    pub fidelity: EvaluationFidelity,
    /// Addressable analytical/surrogate/solver/measurement run receipt.
    pub evidence_ref: String,
    /// Strictly ordered by objective ID when present.
    pub objectives: Vec<ObjectiveObservation>,
    /// Strictly ordered by constraint ID when present.
    pub constraints: Vec<ConstraintObservation>,
    pub status: EvaluationStatus,
}

impl CandidateEvaluation {
    pub fn validate(&self, profile: &OptimizationProfile) -> Result<(), OptimizationError> {
        profile.validate()?;
        self.candidate.validate(profile)?;
        let expected_digest = profile.digest()?;
        if self.profile_digest != expected_digest {
            return Err(OptimizationError::ProfileDigestMismatch);
        }
        self.fidelity.validate()?;
        require_id("evaluation.evidence_ref", &self.evidence_ref)?;
        ensure_strictly_ordered(
            "evaluation.objectives",
            self.objectives.iter().map(|value| value.objective_id.as_str()),
        )?;
        ensure_strictly_ordered(
            "evaluation.constraints",
            self.constraints.iter().map(|value| value.constraint_id.as_str()),
        )?;

        let objective_specs: BTreeMap<_, _> = profile
            .objectives
            .iter()
            .map(|spec| (spec.id.as_str(), spec))
            .collect();
        for observation in &self.objectives {
            let spec = objective_specs
                .get(observation.objective_id.as_str())
                .ok_or_else(|| OptimizationError::UnknownObjective(observation.objective_id.clone()))?;
            validate_observed_scalar(
                "objective.value",
                observation.value,
                &observation.unit,
                &spec.unit,
                observation.uncertainty,
            )?;
        }

        let constraint_specs: BTreeMap<_, _> = profile
            .hard_constraints
            .iter()
            .map(|spec| (spec.id.as_str(), spec))
            .collect();
        for observation in &self.constraints {
            let spec = constraint_specs
                .get(observation.constraint_id.as_str())
                .ok_or_else(|| OptimizationError::UnknownConstraint(observation.constraint_id.clone()))?;
            validate_observed_scalar(
                "constraint.value",
                observation.value,
                &observation.unit,
                &spec.unit,
                None,
            )?;
            let actual = constraint_satisfied(observation.value, &spec.rule);
            if actual != observation.satisfied {
                return Err(OptimizationError::ConstraintSatisfactionMismatch {
                    constraint_id: observation.constraint_id.clone(),
                });
            }
        }

        match &self.status {
            EvaluationStatus::Complete => {
                if self.objectives.len() != profile.objectives.len() {
                    return Err(OptimizationError::IncompleteObjectives);
                }
                if self.constraints.len() != profile.hard_constraints.len() {
                    return Err(OptimizationError::IncompleteConstraints);
                }
                if self.constraints.iter().any(|constraint| !constraint.satisfied) {
                    return Err(OptimizationError::CompleteButInfeasible);
                }
            }
            EvaluationStatus::Infeasible => {
                if self.constraints.is_empty() || self.constraints.iter().all(|value| value.satisfied) {
                    return Err(OptimizationError::InfeasibleWithoutFailedConstraint);
                }
            }
            EvaluationStatus::Failed { reason } => {
                if reason.trim().is_empty() {
                    return Err(OptimizationError::EmptyFailureReason);
                }
                if !self.objectives.is_empty() || !self.constraints.is_empty() {
                    return Err(OptimizationError::FailedEvaluationHasMetrics);
                }
            }
        }
        Ok(())
    }
}

impl DesignCandidate {
    pub fn validate(&self, profile: &OptimizationProfile) -> Result<(), OptimizationError> {
        require_id("candidate.id", &self.id)?;
        ensure_strictly_ordered(
            "candidate.assignments",
            self.assignments.iter().map(|value| value.variable_id.as_str()),
        )?;
        if self.assignments.len() != profile.variables.len() {
            return Err(OptimizationError::CandidateDoesNotCoverProfile);
        }
        for (assignment, variable) in self.assignments.iter().zip(&profile.variables) {
            if assignment.variable_id != variable.id {
                return Err(OptimizationError::CandidateDoesNotCoverProfile);
            }
            validate_assignment(assignment, variable)?;
        }
        Ok(())
    }
}

/// Result of a pairwise Pareto comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dominance {
    LeftDominates,
    RightDominates,
    Equivalent,
    Tradeoff,
    IncomparableFidelity,
    Unqualified,
}

/// Compare two evaluations under the same optimization profile.
///
/// Cross-fidelity comparison is deliberately not promoted to Pareto dominance.
/// Optimizers may use surrogate scores to choose *what to evaluate next*, but a
/// surrogate candidate cannot become numerically/measured Pareto-superior solely
/// by comparing unlike authority classes.
pub fn compare_dominance(
    left: &CandidateEvaluation,
    right: &CandidateEvaluation,
    profile: &OptimizationProfile,
) -> Result<Dominance, OptimizationError> {
    left.validate(profile)?;
    right.validate(profile)?;

    if left.fidelity.class() != right.fidelity.class() {
        return Ok(Dominance::IncomparableFidelity);
    }

    match (&left.status, &right.status) {
        (EvaluationStatus::Failed { .. }, _) | (_, EvaluationStatus::Failed { .. }) => {
            return Ok(Dominance::Unqualified);
        }
        (EvaluationStatus::Complete, EvaluationStatus::Infeasible) => {
            return Ok(Dominance::LeftDominates);
        }
        (EvaluationStatus::Infeasible, EvaluationStatus::Complete) => {
            return Ok(Dominance::RightDominates);
        }
        (EvaluationStatus::Infeasible, EvaluationStatus::Infeasible) => {
            return Ok(Dominance::Unqualified);
        }
        (EvaluationStatus::Complete, EvaluationStatus::Complete) => {}
    }

    let left_values: BTreeMap<_, _> = left
        .objectives
        .iter()
        .map(|value| (value.objective_id.as_str(), value.value))
        .collect();
    let right_values: BTreeMap<_, _> = right
        .objectives
        .iter()
        .map(|value| (value.objective_id.as_str(), value.value))
        .collect();

    let mut left_no_worse = true;
    let mut right_no_worse = true;
    let mut left_strict = false;
    let mut right_strict = false;

    for objective in &profile.objectives {
        let left_value = *left_values
            .get(objective.id.as_str())
            .ok_or(OptimizationError::IncompleteObjectives)?;
        let right_value = *right_values
            .get(objective.id.as_str())
            .ok_or(OptimizationError::IncompleteObjectives)?;
        match objective.direction {
            ObjectiveDirection::Minimize => {
                left_no_worse &= left_value <= right_value;
                right_no_worse &= right_value <= left_value;
                left_strict |= left_value < right_value;
                right_strict |= right_value < left_value;
            }
            ObjectiveDirection::Maximize => {
                left_no_worse &= left_value >= right_value;
                right_no_worse &= right_value >= left_value;
                left_strict |= left_value > right_value;
                right_strict |= right_value > left_value;
            }
        }
    }

    Ok(if left_no_worse && left_strict {
        Dominance::LeftDominates
    } else if right_no_worse && right_strict {
        Dominance::RightDominates
    } else if left_no_worse && right_no_worse {
        Dominance::Equivalent
    } else {
        Dominance::Tradeoff
    })
}

/// Descriptive same-fidelity Pareto frontier indices.
///
/// Failed and infeasible evaluations are excluded. Mixing authority classes in
/// one canonical frontier is rejected rather than silently blending them.
pub fn pareto_frontier(
    evaluations: &[CandidateEvaluation],
    profile: &OptimizationProfile,
) -> Result<Vec<usize>, OptimizationError> {
    profile.validate()?;
    let mut fidelity_class = None;
    let mut qualified = Vec::new();
    for (index, evaluation) in evaluations.iter().enumerate() {
        evaluation.validate(profile)?;
        if !matches!(evaluation.status, EvaluationStatus::Complete) {
            continue;
        }
        match fidelity_class {
            None => fidelity_class = Some(evaluation.fidelity.class()),
            Some(existing) if existing != evaluation.fidelity.class() => {
                return Err(OptimizationError::MixedFidelityFrontier);
            }
            _ => {}
        }
        qualified.push(index);
    }

    let mut frontier = Vec::new();
    'candidate: for &index in &qualified {
        for &other in &qualified {
            if index == other {
                continue;
            }
            if compare_dominance(&evaluations[other], &evaluations[index], profile)?
                == Dominance::LeftDominates
            {
                continue 'candidate;
            }
        }
        frontier.push(index);
    }
    Ok(frontier)
}

fn validate_domain(domain: &VariableDomain) -> Result<(), OptimizationError> {
    match domain {
        VariableDomain::Continuous { lower, upper } => {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(OptimizationError::InvalidContinuousDomain {
                    lower: *lower,
                    upper: *upper,
                });
            }
        }
        VariableDomain::Discrete { allowed } => {
            if allowed.is_empty() {
                return Err(OptimizationError::EmptyCollection("variable.allowed"));
            }
            if allowed.windows(2).any(|window| window[0] >= window[1]) {
                return Err(OptimizationError::NonCanonicalOrder("variable.allowed"));
            }
        }
        VariableDomain::Categorical { allowed } => {
            if allowed.is_empty() {
                return Err(OptimizationError::EmptyCollection("variable.allowed"));
            }
            if allowed.iter().any(|value| value.trim().is_empty()) {
                return Err(OptimizationError::EmptyIdentifier("variable.category"));
            }
            if allowed.windows(2).any(|window| window[0] >= window[1]) {
                return Err(OptimizationError::NonCanonicalOrder("variable.allowed"));
            }
        }
    }
    Ok(())
}

fn validate_assignment(
    assignment: &DesignAssignment,
    variable: &DesignVariable,
) -> Result<(), OptimizationError> {
    match (&assignment.value, &variable.domain) {
        (DesignValue::Continuous(value), VariableDomain::Continuous { lower, upper }) => {
            if !value.is_finite() || value < lower || value > upper {
                return Err(OptimizationError::AssignmentOutsideDomain(
                    assignment.variable_id.clone(),
                ));
            }
        }
        (DesignValue::Discrete(value), VariableDomain::Discrete { allowed }) => {
            if !allowed.contains(value) {
                return Err(OptimizationError::AssignmentOutsideDomain(
                    assignment.variable_id.clone(),
                ));
            }
        }
        (DesignValue::Categorical(value), VariableDomain::Categorical { allowed }) => {
            if !allowed.contains(value) {
                return Err(OptimizationError::AssignmentOutsideDomain(
                    assignment.variable_id.clone(),
                ));
            }
        }
        _ => {
            return Err(OptimizationError::AssignmentKindMismatch(
                assignment.variable_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_constraint_rule(rule: &ConstraintRule) -> Result<(), OptimizationError> {
    match *rule {
        ConstraintRule::AtMost(value) | ConstraintRule::AtLeast(value) => {
            if !value.is_finite() {
                return Err(OptimizationError::NonFiniteBound);
            }
        }
        ConstraintRule::BetweenInclusive { lower, upper } => {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(OptimizationError::InvalidConstraintInterval { lower, upper });
            }
        }
    }
    Ok(())
}

fn constraint_satisfied(value: f64, rule: &ConstraintRule) -> bool {
    match *rule {
        ConstraintRule::AtMost(maximum) => value <= maximum,
        ConstraintRule::AtLeast(minimum) => value >= minimum,
        ConstraintRule::BetweenInclusive { lower, upper } => value >= lower && value <= upper,
    }
}

fn validate_observed_scalar(
    field: &'static str,
    value: f64,
    unit: &str,
    expected_unit: &str,
    uncertainty: Option<UncertaintyEstimate>,
) -> Result<(), OptimizationError> {
    if !value.is_finite() {
        return Err(OptimizationError::NonFiniteObservation(field));
    }
    if unit != expected_unit {
        return Err(OptimizationError::UnitMismatch {
            expected: expected_unit.to_string(),
            actual: unit.to_string(),
        });
    }
    if let Some(uncertainty) = uncertainty {
        if !uncertainty.epistemic.is_finite()
            || !uncertainty.aleatoric.is_finite()
            || !(0.0..=1.0).contains(&uncertainty.epistemic)
            || !(0.0..=1.0).contains(&uncertainty.aleatoric)
        {
            return Err(OptimizationError::InvalidUncertainty);
        }
        if let Some(interval) = uncertainty.interval {
            if !interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower > interval.upper
                || !interval.contains(value)
            {
                return Err(OptimizationError::InvalidUncertainty);
            }
        }
    }
    Ok(())
}

fn require_id(field: &'static str, value: &str) -> Result<(), OptimizationError> {
    if value.trim().is_empty() {
        Err(OptimizationError::EmptyIdentifier(field))
    } else {
        Ok(())
    }
}

fn ensure_strictly_ordered<'a>(
    field: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), OptimizationError> {
    let values: Vec<_> = values.collect();
    if values.windows(2).any(|window| window[0] >= window[1]) {
        return Err(OptimizationError::NonCanonicalOrder(field));
    }
    Ok(())
}

fn push_field(material: &mut String, key: &str, value: &str) {
    material.push_str(&key.len().to_string());
    material.push(':');
    material.push_str(key);
    material.push('=');
    material.push_str(&value.len().to_string());
    material.push(':');
    material.push_str(value);
    material.push(';');
}

fn push_f64(material: &mut String, key: &str, value: f64) {
    push_field(material, key, &format!("{:016x}", value.to_bits()));
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum OptimizationError {
    #[error("identifier {0} cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("collection {0} cannot be empty")]
    EmptyCollection(&'static str),
    #[error("{0} must be strictly ordered and duplicate-free")]
    NonCanonicalOrder(&'static str),
    #[error("invalid continuous domain [{lower}, {upper}]")]
    InvalidContinuousDomain { lower: f64, upper: f64 },
    #[error("constraint bound must be finite")]
    NonFiniteBound,
    #[error("invalid constraint interval [{lower}, {upper}]")]
    InvalidConstraintInterval { lower: f64, upper: f64 },
    #[error("candidate does not provide exactly the profile variable set")]
    CandidateDoesNotCoverProfile,
    #[error("assignment kind does not match domain for {0}")]
    AssignmentKindMismatch(String),
    #[error("assignment is outside admitted domain for {0}")]
    AssignmentOutsideDomain(String),
    #[error("evaluation profile digest does not match the supplied profile")]
    ProfileDigestMismatch,
    #[error("unknown objective {0}")]
    UnknownObjective(String),
    #[error("unknown constraint {0}")]
    UnknownConstraint(String),
    #[error("observation {0} must be finite")]
    NonFiniteObservation(&'static str),
    #[error("unit mismatch: expected {expected}, got {actual}")]
    UnitMismatch { expected: String, actual: String },
    #[error("invalid uncertainty metadata")]
    InvalidUncertainty,
    #[error("constraint {constraint_id} satisfaction flag disagrees with its rule")]
    ConstraintSatisfactionMismatch { constraint_id: String },
    #[error("complete evaluation is missing one or more objective metrics")]
    IncompleteObjectives,
    #[error("complete evaluation is missing one or more constraint metrics")]
    IncompleteConstraints,
    #[error("evaluation marked complete contains a failed hard constraint")]
    CompleteButInfeasible,
    #[error("evaluation marked infeasible has no failed hard constraint")]
    InfeasibleWithoutFailedConstraint,
    #[error("failed evaluation requires a non-empty reason")]
    EmptyFailureReason,
    #[error("failed evaluation cannot carry promotable metrics")]
    FailedEvaluationHasMetrics,
    #[error("canonical Pareto frontier cannot mix evidence fidelity classes")]
    MixedFidelityFrontier,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> OptimizationProfile {
        OptimizationProfile {
            id: "speaker-reference-v1".into(),
            variables: vec![
                DesignVariable {
                    id: "box_volume".into(),
                    unit: Some("m^3".into()),
                    domain: VariableDomain::Continuous {
                        lower: 0.005,
                        upper: 0.05,
                    },
                },
                DesignVariable {
                    id: "woofer_count".into(),
                    unit: None,
                    domain: VariableDomain::Discrete {
                        allowed: vec![1, 2, 4],
                    },
                },
            ],
            objectives: vec![
                ObjectiveSpec {
                    id: "cost".into(),
                    unit: "USD".into(),
                    direction: ObjectiveDirection::Minimize,
                },
                ObjectiveSpec {
                    id: "max_spl".into(),
                    unit: "dB".into(),
                    direction: ObjectiveDirection::Maximize,
                },
            ],
            hard_constraints: vec![HardConstraintSpec {
                id: "coil_temp_limit".into(),
                metric_id: "coil_temp".into(),
                unit: "degC".into(),
                rule: ConstraintRule::AtMost(180.0),
            }],
        }
    }

    fn candidate(id: &str, volume: f64, count: i64) -> DesignCandidate {
        DesignCandidate {
            id: id.into(),
            assignments: vec![
                DesignAssignment {
                    variable_id: "box_volume".into(),
                    value: DesignValue::Continuous(volume),
                },
                DesignAssignment {
                    variable_id: "woofer_count".into(),
                    value: DesignValue::Discrete(count),
                },
            ],
        }
    }

    fn complete(
        profile: &OptimizationProfile,
        id: &str,
        cost: f64,
        spl: f64,
        fidelity: EvaluationFidelity,
    ) -> CandidateEvaluation {
        CandidateEvaluation {
            candidate: candidate(id, 0.02, 2),
            profile_digest: profile.digest().unwrap(),
            fidelity,
            evidence_ref: format!("evidence:{id}"),
            objectives: vec![
                ObjectiveObservation {
                    objective_id: "cost".into(),
                    value: cost,
                    unit: "USD".into(),
                    uncertainty: None,
                },
                ObjectiveObservation {
                    objective_id: "max_spl".into(),
                    value: spl,
                    unit: "dB".into(),
                    uncertainty: None,
                },
            ],
            constraints: vec![ConstraintObservation {
                constraint_id: "coil_temp_limit".into(),
                value: 120.0,
                unit: "degC".into(),
                satisfied: true,
            }],
            status: EvaluationStatus::Complete,
        }
    }

    #[test]
    fn profile_digest_binds_objective_semantics() {
        let a = profile();
        let mut b = a.clone();
        b.objectives[0].direction = ObjectiveDirection::Maximize;
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn candidate_must_cover_exact_canonical_variable_set() {
        let profile = profile();
        let mut candidate = candidate("candidate-a", 0.02, 2);
        candidate.assignments.reverse();
        assert!(matches!(
            candidate.validate(&profile),
            Err(OptimizationError::NonCanonicalOrder("candidate.assignments"))
        ));
    }

    #[test]
    fn complete_evaluation_cannot_hide_hard_constraint_failure() {
        let profile = profile();
        let mut evaluation = complete(
            &profile,
            "candidate-a",
            1000.0,
            110.0,
            EvaluationFidelity::Numerical {
                solver_id: "solver-a".into(),
            },
        );
        evaluation.constraints[0].value = 200.0;
        evaluation.constraints[0].satisfied = false;
        assert_eq!(
            evaluation.validate(&profile),
            Err(OptimizationError::CompleteButInfeasible)
        );
    }

    #[test]
    fn missing_objective_fails_closed() {
        let profile = profile();
        let mut evaluation = complete(
            &profile,
            "candidate-a",
            1000.0,
            110.0,
            EvaluationFidelity::Analytical {
                model_id: "model-a".into(),
            },
        );
        evaluation.objectives.pop();
        assert_eq!(
            evaluation.validate(&profile),
            Err(OptimizationError::IncompleteObjectives)
        );
    }

    #[test]
    fn surrogate_and_numerical_results_are_not_pareto_comparable() {
        let profile = profile();
        let surrogate = complete(
            &profile,
            "surrogate",
            800.0,
            115.0,
            EvaluationFidelity::Surrogate {
                model_id: "surrogate-v1".into(),
            },
        );
        let numerical = complete(
            &profile,
            "numerical",
            1000.0,
            110.0,
            EvaluationFidelity::Numerical {
                solver_id: "fem-v1".into(),
            },
        );
        assert_eq!(
            compare_dominance(&surrogate, &numerical, &profile).unwrap(),
            Dominance::IncomparableFidelity
        );
        assert_eq!(
            pareto_frontier(&[surrogate, numerical], &profile),
            Err(OptimizationError::MixedFidelityFrontier)
        );
    }

    #[test]
    fn hard_constraint_failure_is_not_rescued_by_better_objectives() {
        let profile = profile();
        let feasible = complete(
            &profile,
            "feasible",
            1200.0,
            108.0,
            EvaluationFidelity::Numerical {
                solver_id: "solver".into(),
            },
        );
        let mut infeasible = complete(
            &profile,
            "infeasible",
            500.0,
            120.0,
            EvaluationFidelity::Numerical {
                solver_id: "solver".into(),
            },
        );
        infeasible.constraints[0].value = 210.0;
        infeasible.constraints[0].satisfied = false;
        infeasible.status = EvaluationStatus::Infeasible;
        assert_eq!(
            compare_dominance(&feasible, &infeasible, &profile).unwrap(),
            Dominance::LeftDominates
        );
    }

    #[test]
    fn pareto_frontier_preserves_tradeoffs() {
        let profile = profile();
        let fidelity = EvaluationFidelity::Numerical {
            solver_id: "solver".into(),
        };
        let cheap = complete(&profile, "cheap", 800.0, 105.0, fidelity.clone());
        let loud = complete(&profile, "loud", 1200.0, 115.0, fidelity.clone());
        let dominated = complete(&profile, "dominated", 1500.0, 104.0, fidelity);
        assert_eq!(
            pareto_frontier(&[cheap, loud, dominated], &profile).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn failed_evaluation_cannot_carry_favorable_metrics() {
        let profile = profile();
        let mut failed = complete(
            &profile,
            "failed",
            1.0,
            200.0,
            EvaluationFidelity::Numerical {
                solver_id: "solver".into(),
            },
        );
        failed.status = EvaluationStatus::Failed {
            reason: "solver did not converge".into(),
        };
        assert_eq!(
            failed.validate(&profile),
            Err(OptimizationError::FailedEvaluationHasMetrics)
        );
    }
}
