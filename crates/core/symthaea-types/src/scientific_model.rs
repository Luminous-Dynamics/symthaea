// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root.
//
// Solver-independent scientific model / physics IR.
//
// This module deliberately lives below solver adapters. It describes an
// explicit scientific hypothesis without importing Lanyon, Lean, a numerical
// solver, or the physics-bridge AST. Backend projections must consume this
// representation and must not manufacture semantic meaning that is absent here.

use serde::{Deserialize, Serialize};

use crate::{ModelMaturity, PhysicalType, QuantityKind, ScalarDomain, TypeJudgement};

pub const SCIENTIFIC_MODEL_SCHEMA: &str = "SCIENTIFIC_MODEL.v0.1";

/// A small, solver-independent expression language for canonical scientific
/// models. It is intentionally narrower than the physics-bridge AST: backend
/// adapters may lower it, but the canonical IR does not depend on them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScientificExpression {
    Symbol(String),
    Rational { numerator: i64, denominator: i64 },
    Add(Vec<Self>),
    Multiply(Vec<Self>),
    Negate(Box<Self>),
    Derivative {
        expression: Box<Self>,
        with_respect_to: String,
        order: u8,
    },
}

impl ScientificExpression {
    pub fn symbol(name: impl Into<String>) -> Result<Self, String> {
        let name = name.into();
        validate_identifier(&name)?;
        Ok(Self::Symbol(name))
    }

    pub fn rational(numerator: i64, denominator: i64) -> Result<Self, String> {
        if denominator == 0 {
            return Err("scientific-model rational denominator must be non-zero".into());
        }
        Ok(Self::Rational { numerator, denominator })
    }

    fn validate(&self, symbols: &std::collections::BTreeSet<String>) -> Result<(), String> {
        match self {
            Self::Symbol(name) => {
                validate_identifier(name)?;
                if !symbols.contains(name) {
                    return Err(format!("expression references undeclared symbol {name:?}"));
                }
            }
            Self::Rational { denominator, .. } if *denominator != 0 => {}
            Self::Rational { .. } => {
                return Err("scientific-model rational denominator must be non-zero".into())
            }
            Self::Add(terms) | Self::Multiply(terms) => {
                if terms.is_empty() {
                    return Err("scientific-model n-ary expression cannot be empty".into());
                }
                for term in terms {
                    term.validate(symbols)?;
                }
            }
            Self::Negate(expression) => expression.validate(symbols)?,
            Self::Derivative { expression, with_respect_to, order } => {
                validate_identifier(with_respect_to)?;
                if *order == 0 {
                    return Err("derivative order must be greater than zero".into());
                }
                if !symbols.contains(with_respect_to) {
                    return Err(format!(
                        "derivative references undeclared independent symbol {with_respect_to:?}"
                    ));
                }
                expression.validate(symbols)?;
            }
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        // serde_json is used only after validation; all variants are finite,
        // bounded data, so this is a deterministic structural ordering key.
        serde_json::to_string(self).expect("scientific expression serialization is infallible")
    }
}

/// A named physical quantity/field in the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificQuantity {
    pub name: String,
    pub physical_type: PhysicalType,
    pub scalar_domain: ScalarDomain,
    pub role: QuantityRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantityRole {
    Coordinate,
    State,
    Parameter,
    Constant,
    Observable,
    Auxiliary,
}

impl ScientificQuantity {
    pub fn validate(&self) -> Result<(), String> {
        validate_identifier(&self.name)?;
        self.physical_type
            .validate()
            .map_err(|error| format!("quantity {:?}: {}", self.name, error.reason))?;
        if self.physical_type.scalar != self.scalar_domain {
            return Err(format!(
                "quantity {:?} has conflicting scalar domains",
                self.name
            ));
        }
        Ok(())
    }
}

/// An equation is semantic content, not a solver instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificEquation {
    pub name: String,
    pub left: ScientificExpression,
    pub right: ScientificExpression,
}

impl ScientificEquation {
    pub fn validate(
        &self,
        symbols: &std::collections::BTreeSet<String>,
    ) -> Result<(), String> {
        validate_identifier(&self.name)?;
        self.left.validate(symbols)?;
        self.right.validate(symbols)?;
        Ok(())
    }
}

/// Explicit statement of a condition assumed by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAssumption {
    pub id: String,
    pub statement: String,
}

impl ModelAssumption {
    pub fn validate(&self) -> Result<(), String> {
        validate_identifier(&self.id)?;
        if self.statement.trim().is_empty() {
            return Err(format!("assumption {:?} has an empty statement", self.id));
        }
        Ok(())
    }
}

/// Numerical intent is declarative. It does not assert that a requested
/// property was achieved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverIntent {
    pub backend_family: Option<String>,
    pub conservation_required: bool,
    pub positivity_required: bool,
    pub invariant_domain_required: bool,
    pub monotonicity_required: bool,
    pub target_accuracy_order: Option<u8>,
}

impl SolverIntent {
    pub fn validate(&self) -> Result<(), String> {
        if self.target_accuracy_order == Some(0) {
            return Err("solver target accuracy order must be greater than zero".into());
        }
        if let Some(backend) = &self.backend_family {
            if backend.trim().is_empty() {
                return Err("solver backend family cannot be empty when specified".into());
            }
        }
        Ok(())
    }
}

/// A requirement that must be discharged by a future formal, numerical, or
/// empirical evaluator. Status is intentionally absent from the model itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEvidenceRequirement {
    pub id: String,
    pub class: EvidenceClass,
    pub statement: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceClass {
    Formal,
    Numerical,
    Empirical,
}

impl ModelEvidenceRequirement {
    pub fn validate(&self) -> Result<(), String> {
        validate_identifier(&self.id)?;
        if self.statement.trim().is_empty() {
            return Err(format!(
                "evidence requirement {:?} has an empty statement",
                self.id
            ));
        }
        Ok(())
    }
}

/// A deliberate non-claim prevents downstream code from interpreting model
/// identity as a truth certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelNonClaim {
    pub id: String,
    pub statement: String,
}

impl ModelNonClaim {
    pub fn validate(&self) -> Result<(), String> {
        validate_identifier(&self.id)?;
        if self.statement.trim().is_empty() {
            return Err(format!("non-claim {:?} has an empty statement", self.id));
        }
        Ok(())
    }
}

/// Solver-independent canonical scientific model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificModel {
    pub schema_revision: String,
    pub model_id: String,
    pub domain: String,
    pub maturity: ModelMaturity,
    pub quantities: Vec<ScientificQuantity>,
    pub equations: Vec<ScientificEquation>,
    pub assumptions: Vec<ModelAssumption>,
    pub solver_intent: SolverIntent,
    pub evidence_requirements: Vec<ModelEvidenceRequirement>,
    pub non_claims: Vec<ModelNonClaim>,
    pub provenance: Vec<String>,
}

impl ScientificModel {
    pub fn new(
        model_id: impl Into<String>,
        domain: impl Into<String>,
        maturity: ModelMaturity,
        mut quantities: Vec<ScientificQuantity>,
        mut equations: Vec<ScientificEquation>,
        mut assumptions: Vec<ModelAssumption>,
        solver_intent: SolverIntent,
        mut evidence_requirements: Vec<ModelEvidenceRequirement>,
        mut non_claims: Vec<ModelNonClaim>,
        mut provenance: Vec<String>,
    ) -> Result<Self, String> {
        let model = Self {
            schema_revision: SCIENTIFIC_MODEL_SCHEMA.into(),
            model_id: model_id.into(),
            domain: domain.into(),
            maturity,
            quantities: {
                quantities.sort_by(|a, b| a.name.cmp(&b.name));
                quantities
            },
            equations: {
                equations.sort_by(|a, b| a.name.cmp(&b.name));
                equations
            },
            assumptions: {
                assumptions.sort_by(|a, b| a.id.cmp(&b.id));
                assumptions
            },
            solver_intent,
            evidence_requirements: {
                evidence_requirements.sort_by(|a, b| a.id.cmp(&b.id));
                evidence_requirements
            },
            non_claims: {
                non_claims.sort_by(|a, b| a.id.cmp(&b.id));
                non_claims
            },
            provenance: {
                provenance.sort();
                provenance
            },
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_revision != SCIENTIFIC_MODEL_SCHEMA {
            return Err("unsupported scientific-model schema revision".into());
        }
        validate_identifier(&self.model_id)?;
        if self.domain.trim().is_empty() {
            return Err("scientific-model domain cannot be empty".into());
        }

        let mut symbols = std::collections::BTreeSet::new();
        for quantity in &self.quantities {
            quantity.validate()?;
            if !symbols.insert(quantity.name.clone()) {
                return Err(format!("duplicate scientific quantity {:?}", quantity.name));
            }
        }

        reject_duplicate_ids(
            self.equations.iter().map(|entry| &entry.name),
            "equation",
        )?;
        reject_duplicate_ids(
            self.assumptions.iter().map(|entry| &entry.id),
            "assumption",
        )?;
        reject_duplicate_ids(
            self.evidence_requirements.iter().map(|entry| &entry.id),
            "evidence requirement",
        )?;
        reject_duplicate_ids(
            self.non_claims.iter().map(|entry| &entry.id),
            "non-claim",
        )?;

        for equation in &self.equations {
            equation.validate(&symbols)?;
        }
        for assumption in &self.assumptions {
            assumption.validate()?;
        }
        self.solver_intent.validate()?;
        for requirement in &self.evidence_requirements {
            requirement.validate()?;
        }
        for non_claim in &self.non_claims {
            non_claim.validate()?;
        }
        if self.provenance.iter().any(|entry| entry.trim().is_empty()) {
            return Err("scientific-model provenance entries cannot be empty".into());
        }

        // A model must explicitly state that formal/numerical/empirical
        // evidence is still a separate concern. This is a semantic firewall,
        // not an evaluator result.
        if self.non_claims.is_empty() {
            return Err(
                "scientific model must declare at least one explicit non-claim".into(),
            );
        }

        Ok(())
    }

    pub fn type_judgment(&self) -> TypeJudgement<()> {
        match self.validate() {
            Ok(()) => TypeJudgement::Valid(()),
            Err(reason) => TypeJudgement::Invalid(crate::PhysicalTypeError {
                operation: "scientific_model".into(),
                reason,
            }),
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ScientificModel serialization is infallible")
    }

    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}

fn validate_identifier(value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err("scientific-model identifiers cannot be empty".into());
    }
    if value
        .chars()
        .any(|character| character.is_whitespace() || character == '/' || character == '\\')
    {
        return Err(format!("invalid scientific-model identifier {value:?}"));
    }
    Ok(())
}

fn reject_duplicate_ids<'a, I>(values: I, label: &str) -> Result<(), String>
where
    I: IntoIterator<Item = &'a String>,
{
    let mut previous: Option<&str> = None;
    for value in values {
        if previous == Some(value.as_str()) {
            return Err(format!("duplicate {label} {:?}", value));
        }
        previous = Some(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicalDimension, QuantityKind};

    fn velocity(name: &str) -> ScientificQuantity {
        let mut physical_type =
            PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY);
        physical_type.scalar = ScalarDomain::Real;
        ScientificQuantity {
            name: name.into(),
            physical_type,
            scalar_domain: ScalarDomain::Real,
            role: QuantityRole::State,
        }
    }

    fn non_claim() -> ModelNonClaim {
        ModelNonClaim {
            id: "not-empirical-truth".into(),
            statement: "Formal or numerical success does not establish empirical truth.".into(),
        }
    }

    fn model() -> ScientificModel {
        ScientificModel::new(
            "burgers-v0-1",
            "fluid-dynamics",
            ModelMaturity::ResearchPrototype,
            vec![
                velocity("u"),
                ScientificQuantity {
                    name: "x".into(),
                    physical_type: PhysicalType::with_kind(
                        QuantityKind::Length,
                        PhysicalDimension::LENGTH,
                    ),
                    scalar_domain: ScalarDomain::Real,
                    role: QuantityRole::Coordinate,
                },
                ScientificQuantity {
                    name: "t".into(),
                    physical_type: PhysicalType::with_kind(
                        QuantityKind::Time,
                        PhysicalDimension::TIME,
                    ),
                    scalar_domain: ScalarDomain::Real,
                    role: QuantityRole::Coordinate,
                },
            ],
            vec![ScientificEquation {
                name: "burgers".into(),
                left: ScientificExpression::Add(vec![
                    ScientificExpression::Derivative {
                        expression: Box::new(ScientificExpression::symbol("u").unwrap()),
                        with_respect_to: "t".into(),
                        order: 1,
                    },
                    ScientificExpression::Multiply(vec![
                        ScientificExpression::symbol("u").unwrap(),
                        ScientificExpression::Derivative {
                            expression: Box::new(ScientificExpression::symbol("u").unwrap()),
                            with_respect_to: "x".into(),
                            order: 1,
                        },
                    ]),
                ]),
                right: ScientificExpression::rational(0, 1).unwrap(),
            }],
            vec![ModelAssumption {
                id: "smooth-before-shock".into(),
                statement: "The represented regime is evaluated before unresolved shock structure.".into(),
            }],
            SolverIntent {
                backend_family: Some("finite-volume".into()),
                conservation_required: true,
                positivity_required: false,
                invariant_domain_required: false,
                monotonicity_required: true,
                target_accuracy_order: Some(1),
            },
            vec![ModelEvidenceRequirement {
                id: "formal-equivalence".into(),
                class: EvidenceClass::Formal,
                statement: "The backend projection preserves the declared equation structure.".into(),
            }],
            vec![non_claim()],
            vec!["fixture:burgers-v0-1".into()],
        )
        .unwrap()
    }

    #[test]
    fn canonical_digest_is_stable() {
        let first = model();
        let second = model();
        assert_eq!(first.canonical_bytes(), second.canonical_bytes());
        assert_eq!(first.digest(), second.digest());
    }

    #[test]
    fn quantity_order_does_not_change_identity() {
        let first = model();
        let mut second = first.clone();
        second.quantities.reverse();
        assert_ne!(first.canonical_bytes(), second.canonical_bytes());
        // Constructor canonicalization, rather than post-hoc mutation, is the
        // identity-preserving path.
        let rebuilt = ScientificModel::new(
            second.model_id.clone(),
            second.domain.clone(),
            second.maturity,
            second.quantities,
            second.equations,
            second.assumptions,
            second.solver_intent,
            second.evidence_requirements,
            second.non_claims,
            second.provenance,
        )
        .unwrap();
        assert_eq!(first.digest(), rebuilt.digest());
    }

    #[test]
    fn changing_assumption_changes_identity() {
        let first = model();
        let mut assumptions = first.assumptions.clone();
        assumptions[0].statement.push_str(" Revised.");
        let changed = ScientificModel::new(
            first.model_id.clone(),
            first.domain.clone(),
            first.maturity,
            first.quantities.clone(),
            first.equations.clone(),
            assumptions,
            first.solver_intent.clone(),
            first.evidence_requirements.clone(),
            first.non_claims.clone(),
            first.provenance.clone(),
        )
        .unwrap();
        assert_ne!(first.digest(), changed.digest());
    }

    #[test]
    fn undeclared_symbols_fail_closed() {
        let mut broken = model();
        broken.equations[0].left = ScientificExpression::symbol("unknown").unwrap();
        assert!(broken.validate().is_err());
    }

    #[test]
    fn unknown_physical_type_does_not_become_dimensionless() {
        let quantity = ScientificQuantity {
            name: "u".into(),
            physical_type: PhysicalType::unknown(),
            scalar_domain: ScalarDomain::Unknown,
            role: QuantityRole::State,
        };
        assert!(quantity.validate().is_ok());
        assert_eq!(quantity.physical_type.dimension, None);
    }

    #[test]
    fn non_claim_is_required() {
        let mut broken = model();
        broken.non_claims.clear();
        assert!(broken.validate().is_err());
    }
}
