//! Versioned hybrid-dynamics extension for the canonical ScientificModel IR.
//!
//! This module contains physical semantics only. Numerical event-time
//! enclosures, branch certificates, posterior probabilities and solver run
//! receipts remain separate evidence artifacts.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::scientific_model::{
    validate_identifier, ScientificEquation, ScientificExpression, ScientificModel,
    ScientificQuantity,
};
use crate::{
    PhysicalDimension, PhysicalType, QuantityKind, QuantityRole, ScalarDomain, TypeJudgement,
};

pub const HYBRID_DYNAMICS_SCHEMA_V0_1: &str = "HYBRID_DYNAMICS.v0.1";

/// Direction in which a trajectory crosses a scalar guard surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardCrossingDirection {
    NegativeToPositive,
    PositiveToNegative,
    EitherDirection,
    GrazingAnalysisOnly,
}

/// Policy for event-time overlap. Neither policy invents a total event order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimultaneousEventPolicy {
    /// Overlapping event-time enclosures remain unresolved until a joint law
    /// and independently verified joint-event certificate are available.
    RequireJointEventCertificate,
    /// Preserve alternative event-order hypotheses for downstream solvers.
    PreserveUnresolvedOrderAlternatives,
}

/// Exact rational value used in parameter-domain conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RationalValue {
    pub numerator: i64,
    pub denominator: i64,
}

impl RationalValue {
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, String> {
        if denominator <= 0 {
            return Err("hybrid rational denominator must be positive".into());
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator as u64) as i64;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

/// A declared condition on a physical parameter that selects a transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterRelation {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterCondition {
    pub parameter_id: String,
    pub relation: ParameterRelation,
    pub value: RationalValue,
}

impl ParameterCondition {
    fn validate(&self) -> Result<(), String> {
        validate_identifier(&self.parameter_id)?;
        if self.value.denominator <= 0 {
            return Err(format!(
                "parameter condition {:?} has a non-positive denominator",
                self.parameter_id
            ));
        }
        if RationalValue::new(self.value.numerator, self.value.denominator)? != self.value {
            return Err(format!(
                "parameter condition {:?} must use a reduced rational value",
                self.parameter_id
            ));
        }
        Ok(())
    }
}

/// How a reset relates to its physical impact/contact law.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResetLaw {
    /// Newton normal restitution. The parameter domain must be constrained by
    /// explicit transition conditions; this enum alone does not prove a value
    /// lies in [0, 1].
    NewtonRestitution {
        restitution_parameter_id: String,
    },
    /// A declared constrained-contact law, e.g. a perfectly inelastic resting
    /// contact mode. The law ID is a semantic reference, not a solver result.
    PersistentContact {
        law_id: String,
    },
    /// A separately identified domain-specific reset law.
    Named {
        law_id: String,
    },
}

impl ResetLaw {
    fn validate(&self, quantities: &BTreeMap<&str, &ScientificQuantity>) -> Result<(), String> {
        match self {
            Self::NewtonRestitution {
                restitution_parameter_id,
            } => {
                validate_identifier(restitution_parameter_id)?;
                let quantity = quantities
                    .get(restitution_parameter_id.as_str())
                    .ok_or_else(|| {
                        format!(
                            "Newton restitution references undeclared parameter {:?}",
                            restitution_parameter_id
                        )
                    })?;
                if quantity.role != QuantityRole::Parameter {
                    return Err(format!(
                        "Newton restitution coefficient {:?} is not a model parameter",
                        restitution_parameter_id
                    ));
                }
                if quantity.physical_type.dimension
                    != Some(PhysicalDimension::DIMENSIONLESS)
                    || matches!(
                        quantity.physical_type.scalar,
                        ScalarDomain::Unknown | ScalarDomain::Complex
                    )
                {
                    return Err(format!(
                        "Newton restitution coefficient {:?} must have a known dimensionless type",
                        restitution_parameter_id
                    ));
                }
            }
            Self::PersistentContact { law_id } | Self::Named { law_id } => {
                validate_identifier(law_id)?;
            }
        }
        Ok(())
    }

    fn parameter_id(&self) -> Option<&str> {
        match self {
            Self::NewtonRestitution {
                restitution_parameter_id,
            } => Some(restitution_parameter_id),
            Self::PersistentContact { .. } | Self::Named { .. } => None,
        }
    }
}

/// A discrete dynamical mode with mode-local differential/algebraic equations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridMode {
    pub id: String,
    /// Equations use the existing ScientificExpression language. Differential
    /// equations should place a Derivative expression on the left-hand side.
    /// Algebraic equations may appear in the same mode roster.
    pub equations: Vec<ScientificEquation>,
    /// Every parameter used by this mode's equations must be listed here.
    pub parameter_ids: Vec<String>,
    pub validity_assumption_ids: Vec<String>,
    pub evidence_requirement_ids: Vec<String>,
    pub non_claim_ids: Vec<String>,
}

/// A scalar switching/impact surface. The expression must be dimensionally
/// homogeneous and have a known real scalar domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardSurface {
    pub id: String,
    pub source_mode_id: String,
    pub expression: ScientificExpression,
    pub crossing_direction: GuardCrossingDirection,
    /// Explicit time symbol when the guard depends on time.
    pub time_symbol: Option<String>,
    pub parameter_ids: Vec<String>,
    pub validity_assumption_ids: Vec<String>,
    pub evidence_requirement_ids: Vec<String>,
}

/// One state-coordinate assignment in a reset map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResetAssignment {
    pub target_state_id: String,
    pub expression: ScientificExpression,
}

/// A discrete state reset, including its target mode and physical law identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResetMap {
    pub id: String,
    pub source_mode_id: String,
    pub target_mode_id: String,
    /// Every declared state quantity must be assigned explicitly. Identity
    /// carry-through is expressed as x_plus = x_minus, not as an implicit default.
    pub assignments: Vec<ResetAssignment>,
    pub law: ResetLaw,
    /// Explicit time symbol when reset assignments depend on time.
    pub time_symbol: Option<String>,
    pub parameter_ids: Vec<String>,
    pub validity_assumption_ids: Vec<String>,
    pub evidence_requirement_ids: Vec<String>,
}

/// A transition references a mode, guard and reset without embedding a solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridTransition {
    pub id: String,
    pub source_mode_id: String,
    pub guard_id: String,
    pub reset_id: String,
    pub target_mode_id: String,
    /// Parameter-domain predicates disambiguate cases such as e=0 persistent
    /// contact versus e>0 rebound. Empty means no additional predicate.
    pub parameter_conditions: Vec<ParameterCondition>,
    pub validity_assumption_ids: Vec<String>,
    pub evidence_requirement_ids: Vec<String>,
}

/// Canonical hybrid semantics nested inside a ScientificModel.
///
/// The structure is versioned independently so the parent model can carry an
/// explicit hybrid extension without defining a competing model ontology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridDynamicsV0_1 {
    pub schema_revision: String,
    /// The one physical time coordinate used by all differential equations.
    pub time_symbol: String,
    pub initial_mode_id: String,
    pub modes: Vec<HybridMode>,
    pub guards: Vec<GuardSurface>,
    pub resets: Vec<ResetMap>,
    pub transitions: Vec<HybridTransition>,
    pub simultaneous_event_policy: SimultaneousEventPolicy,
}

impl HybridDynamicsV0_1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        time_symbol: impl Into<String>,
        initial_mode_id: impl Into<String>,
        modes: Vec<HybridMode>,
        guards: Vec<GuardSurface>,
        resets: Vec<ResetMap>,
        transitions: Vec<HybridTransition>,
        simultaneous_event_policy: SimultaneousEventPolicy,
    ) -> Result<Self, String> {
        let mut hybrid = Self {
            schema_revision: HYBRID_DYNAMICS_SCHEMA_V0_1.into(),
            time_symbol: time_symbol.into(),
            initial_mode_id: initial_mode_id.into(),
            modes,
            guards,
            resets,
            transitions,
            simultaneous_event_policy,
        };
        hybrid.canonicalize();
        validate_identifier(&hybrid.time_symbol)?;
        validate_identifier(&hybrid.initial_mode_id)?;
        Ok(hybrid)
    }

    /// Normalize every order-insensitive roster before hashing or comparing
    /// model identity. Physical ordering remains encoded only where the
    /// semantics explicitly require it (e.g. equation expressions).
    pub(crate) fn canonicalize(&mut self) {
        self.modes.sort_by(|a, b| a.id.cmp(&b.id));
        self.guards.sort_by(|a, b| a.id.cmp(&b.id));
        self.resets.sort_by(|a, b| a.id.cmp(&b.id));
        self.transitions.sort_by(|a, b| a.id.cmp(&b.id));

        for mode in &mut self.modes {
            mode.parameter_ids.sort();
            mode.validity_assumption_ids.sort();
            mode.evidence_requirement_ids.sort();
            mode.non_claim_ids.sort();
            mode.equations.sort_by(|a, b| a.name.cmp(&b.name));
        }
        for guard in &mut self.guards {
            guard.parameter_ids.sort();
            guard.validity_assumption_ids.sort();
            guard.evidence_requirement_ids.sort();
        }
        for reset in &mut self.resets {
            reset.assignments.sort_by(|a, b| a.target_state_id.cmp(&b.target_state_id));
            reset.parameter_ids.sort();
            reset.validity_assumption_ids.sort();
            reset.evidence_requirement_ids.sort();
        }
        for transition in &mut self.transitions {
            transition.parameter_conditions.sort_by(|a, b| {
                (
                    a.parameter_id.as_str(),
                    a.relation as u8,
                    a.value.numerator,
                    a.value.denominator,
                )
                    .cmp(&(
                        b.parameter_id.as_str(),
                        b.relation as u8,
                        b.value.numerator,
                        b.value.denominator,
                    ))
            });
            transition.validity_assumption_ids.sort();
            transition.evidence_requirement_ids.sort();
        }
    }

    pub fn validate(&self, model: &ScientificModel) -> Result<(), String> {
        if self.schema_revision != HYBRID_DYNAMICS_SCHEMA_V0_1 {
            return Err("unsupported hybrid-dynamics schema revision".into());
        }
        validate_identifier(&self.time_symbol)?;
        validate_identifier(&self.initial_mode_id)?;

        let quantities: BTreeMap<&str, &ScientificQuantity> = model
            .quantities
            .iter()
            .map(|quantity| (quantity.name.as_str(), quantity))
            .collect();
        let symbols: BTreeSet<String> = quantities.keys().map(|name| (*name).to_string()).collect();
        let assumptions: BTreeSet<&str> =
            model.assumptions.iter().map(|entry| entry.id.as_str()).collect();
        let requirements: BTreeSet<&str> = model
            .evidence_requirements
            .iter()
            .map(|entry| entry.id.as_str())
            .collect();
        let non_claims: BTreeSet<&str> =
            model.non_claims.iter().map(|entry| entry.id.as_str()).collect();

        ensure_unique_ids(self.modes.iter().map(|entry| entry.id.as_str()), "mode")?;
        ensure_unique_ids(self.guards.iter().map(|entry| entry.id.as_str()), "guard")?;
        ensure_unique_ids(self.resets.iter().map(|entry| entry.id.as_str()), "reset")?;
        ensure_unique_ids(
            self.transitions.iter().map(|entry| entry.id.as_str()),
            "transition",
        )?;

        let mode_ids: BTreeSet<&str> = self.modes.iter().map(|entry| entry.id.as_str()).collect();
        let guard_ids: BTreeMap<&str, &GuardSurface> = self
            .guards
            .iter()
            .map(|entry| (entry.id.as_str(), entry))
            .collect();
        let reset_ids: BTreeMap<&str, &ResetMap> = self
            .resets
            .iter()
            .map(|entry| (entry.id.as_str(), entry))
            .collect();

        if !mode_ids.contains(self.initial_mode_id.as_str()) {
            return Err(format!(
                "initial hybrid mode {:?} is not declared",
                self.initial_mode_id
            ));
        }
        let time_quantity = quantities.get(self.time_symbol.as_str()).ok_or_else(|| {
            format!("hybrid time symbol {:?} is undeclared", self.time_symbol)
        })?;
        if time_quantity.role != QuantityRole::Coordinate
            || time_quantity.physical_type.dimension != Some(PhysicalDimension::TIME)
            || matches!(time_quantity.scalar_domain, ScalarDomain::Unknown | ScalarDomain::Complex)
        {
            return Err(format!(
                "hybrid time symbol {:?} must be a known real time coordinate",
                self.time_symbol
            ));
        }
        if self.modes.is_empty()
            || self.guards.is_empty()
            || self.resets.is_empty()
            || self.transitions.is_empty()
        {
            return Err("hybrid dynamics requires modes, guards, resets and transitions".into());
        }

        for mode in &self.modes {
            validate_identifier(&mode.id)?;
            if mode.equations.is_empty() {
                return Err(format!("hybrid mode {:?} has no equations", mode.id));
            }
            validate_parameters(&mode.parameter_ids, &quantities)?;
            validate_references(
                &mode.validity_assumption_ids,
                &assumptions,
                "mode assumption",
            )?;
            validate_references(
                &mode.evidence_requirement_ids,
                &requirements,
                "mode evidence requirement",
            )?;
            validate_references(&mode.non_claim_ids, &non_claims, "mode non-claim")?;
            ensure_unique_ids(
                mode.equations.iter().map(|equation| equation.name.as_str()),
                "mode equation",
            )?;
            for equation in &mode.equations {
                equation.validate(&symbols)?;
                for derivative_time in expression_derivative_variables(&equation.left)
                    .into_iter()
                    .chain(expression_derivative_variables(&equation.right))
                {
                    if derivative_time != self.time_symbol {
                        return Err(format!(
                            "mode {:?} equation {:?} differentiates against {:?}, not the declared hybrid time {:?}",
                            mode.id, equation.name, derivative_time, self.time_symbol
                        ));
                    }
                }
                validate_referenced_parameters(
                    &equation.left,
                    &quantities,
                    &mode.parameter_ids,
                    "mode equation",
                    &equation.name,
                )?;
                validate_referenced_parameters(
                    &equation.right,
                    &quantities,
                    &mode.parameter_ids,
                    "mode equation",
                    &equation.name,
                )?;
                let left = infer_expression_type(&equation.left, &quantities)?
                    .ok_or_else(|| {
                        format!(
                            "mode equation {:?} has an untyped zero left-hand side",
                            equation.name
                        )
                    })?;
                check_expression_type(&equation.right, &left, &quantities).map_err(|reason| {
                    format!(
                        "mode {:?} equation {:?}: {reason}",
                        mode.id, equation.name
                    )
                })?;
            }
        }

        for guard in &self.guards {
            validate_identifier(&guard.id)?;
            if !mode_ids.contains(guard.source_mode_id.as_str()) {
                return Err(format!(
                    "guard {:?} references undeclared source mode {:?}",
                    guard.id, guard.source_mode_id
                ));
            }
            guard.expression.validate(&symbols)?;
            let guard_type = infer_expression_type(&guard.expression, &quantities)?
                .ok_or_else(|| format!("guard {:?} is an untyped zero expression", guard.id))?;
            if guard_type.dimension.is_none()
                || matches!(guard_type.scalar, ScalarDomain::Unknown | ScalarDomain::Complex)
            {
                return Err(format!(
                    "guard {:?} must have a known real physical type",
                    guard.id
                ));
            }
            validate_parameters(&guard.parameter_ids, &quantities)?;
            validate_references(
                &guard.validity_assumption_ids,
                &assumptions,
                "guard validity assumption",
            )?;
            validate_references(
                &guard.evidence_requirement_ids,
                &requirements,
                "guard evidence requirement",
            )?;
            if guard.time_symbol.as_deref().is_some_and(|symbol| symbol != self.time_symbol) {
                return Err(format!(
                    "guard {:?} uses a time symbol other than the declared hybrid time",
                    guard.id
                ));
            }
            validate_time_dependence(
                &guard.expression,
                guard.time_symbol.as_deref(),
                &quantities,
                "guard",
                &guard.id,
            )?;
            validate_referenced_parameters(
                &guard.expression,
                &quantities,
                &guard.parameter_ids,
                "guard",
                &guard.id,
            )?;
        }

        for reset in &self.resets {
            validate_identifier(&reset.id)?;
            if !mode_ids.contains(reset.source_mode_id.as_str())
                || !mode_ids.contains(reset.target_mode_id.as_str())
            {
                return Err(format!(
                    "reset {:?} references an undeclared source or target mode",
                    reset.id
                ));
            }
            if reset.assignments.is_empty() {
                return Err(format!("reset {:?} has no state assignments", reset.id));
            }
            ensure_unique_ids(
                reset.assignments
                    .iter()
                    .map(|assignment| assignment.target_state_id.as_str()),
                "reset target state",
            )?;
            let state_ids: BTreeSet<&str> = model
                .quantities
                .iter()
                .filter(|quantity| quantity.role == QuantityRole::State)
                .map(|quantity| quantity.name.as_str())
                .collect();
            let assigned_ids: BTreeSet<&str> = reset
                .assignments
                .iter()
                .map(|assignment| assignment.target_state_id.as_str())
                .collect();
            if state_ids != assigned_ids {
                return Err(format!(
                    "reset {:?} must explicitly assign every model state exactly once",
                    reset.id
                ));
            }
            for assignment in &reset.assignments {
                validate_identifier(&assignment.target_state_id)?;
                let target = quantities
                    .get(assignment.target_state_id.as_str())
                    .ok_or_else(|| {
                        format!(
                            "reset {:?} targets undeclared state {:?}",
                            reset.id, assignment.target_state_id
                        )
                    })?;
                if target.role != QuantityRole::State {
                    return Err(format!(
                        "reset {:?} target {:?} is not a state quantity",
                        reset.id, assignment.target_state_id
                    ));
                }
                assignment.expression.validate(&symbols)?;
                check_expression_type(&assignment.expression, &target.physical_type, &quantities)
                    .map_err(|reason| {
                        format!(
                            "reset {:?} assignment {:?}: {reason}",
                            reset.id, assignment.target_state_id
                        )
                    })?;
            }
            reset.law.validate(&quantities)?;
            if let Some(parameter_id) = reset.law.parameter_id()
                && !reset.parameter_ids.iter().any(|id| id == parameter_id)
            {
                return Err(format!(
                    "reset {:?} law parameter {:?} is missing from its parameter roster",
                    reset.id, parameter_id
                ));
            }
            validate_parameters(&reset.parameter_ids, &quantities)?;
            validate_references(
                &reset.validity_assumption_ids,
                &assumptions,
                "reset validity assumption",
            )?;
            validate_references(
                &reset.evidence_requirement_ids,
                &requirements,
                "reset evidence requirement",
            )?;
            if reset.time_symbol.as_deref().is_some_and(|symbol| symbol != self.time_symbol) {
                return Err(format!(
                    "reset {:?} uses a time symbol other than the declared hybrid time",
                    reset.id
                ));
            }
            validate_time_dependence(
                &ScientificExpression::Add(
                    reset.assignments
                        .iter()
                        .map(|assignment| assignment.expression.clone())
                        .collect(),
                ),
                reset.time_symbol.as_deref(),
                &quantities,
                "reset",
                &reset.id,
            )?;
            for assignment in &reset.assignments {
                validate_referenced_parameters(
                    &assignment.expression,
                    &quantities,
                    &reset.parameter_ids,
                    "reset",
                    &reset.id,
                )?;
            }
        }

        for transition in &self.transitions {
            validate_identifier(&transition.id)?;
            if !mode_ids.contains(transition.source_mode_id.as_str())
                || !mode_ids.contains(transition.target_mode_id.as_str())
            {
                return Err(format!(
                    "transition {:?} references an undeclared source or target mode",
                    transition.id
                ));
            }
            let guard = guard_ids
                .get(transition.guard_id.as_str())
                .ok_or_else(|| {
                    format!(
                        "transition {:?} references undeclared guard {:?}",
                        transition.id, transition.guard_id
                    )
                })?;
            let reset = reset_ids
                .get(transition.reset_id.as_str())
                .ok_or_else(|| {
                    format!(
                        "transition {:?} references undeclared reset {:?}",
                        transition.id, transition.reset_id
                    )
                })?;
            if guard.source_mode_id != transition.source_mode_id
                || reset.source_mode_id != transition.source_mode_id
                || reset.target_mode_id != transition.target_mode_id
            {
                return Err(format!(
                    "transition {:?} has inconsistent mode/guard/reset references",
                    transition.id
                ));
            }
            validate_references(
                &transition.validity_assumption_ids,
                &assumptions,
                "transition validity assumption",
            )?;
            validate_references(
                &transition.evidence_requirement_ids,
                &requirements,
                "transition evidence requirement",
            )?;
            if let ResetLaw::NewtonRestitution {
                restitution_parameter_id,
            } = &reset.law
            {
                let zero = RationalValue {
                    numerator: 0,
                    denominator: 1,
                };
                let one = RationalValue {
                    numerator: 1,
                    denominator: 1,
                };
                let has_strict_positive_lower_bound = transition.parameter_conditions.iter().any(
                    |condition| {
                        condition.parameter_id.as_str() == restitution_parameter_id.as_str()
                            && condition.relation == ParameterRelation::GreaterThan
                            && condition.value == zero
                    },
                );
                let has_unit_upper_bound = transition.parameter_conditions.iter().any(
                    |condition| {
                        condition.parameter_id.as_str() == restitution_parameter_id.as_str()
                            && condition.relation == ParameterRelation::LessOrEqual
                            && condition.value == one
                    },
                );
                if !has_strict_positive_lower_bound || !has_unit_upper_bound {
                    return Err(format!(
                        "Newton restitution transition {:?} must explicitly constrain {:?} to 0 < e <= 1; e=0 requires a distinct contact reset",
                        transition.id, restitution_parameter_id
                    ));
                }
            }

            let mut condition_keys = BTreeSet::new();
            for condition in &transition.parameter_conditions {
                condition.validate()?;
                if !condition_keys.insert((
                    condition.parameter_id.as_str(),
                    condition.relation as u8,
                    condition.value.numerator,
                    condition.value.denominator,
                )) {
                    return Err(format!(
                        "transition {:?} has a duplicate parameter condition",
                        transition.id
                    ));
                }
                let quantity = quantities
                    .get(condition.parameter_id.as_str())
                    .ok_or_else(|| {
                        format!(
                            "transition {:?} condition references undeclared parameter {:?}",
                            transition.id, condition.parameter_id
                        )
                    })?;
                if quantity.role != QuantityRole::Parameter {
                    return Err(format!(
                        "transition {:?} condition target {:?} is not a parameter",
                        transition.id, condition.parameter_id
                    ));
                }
                if quantity.physical_type.dimension
                    != Some(PhysicalDimension::DIMENSIONLESS)
                    || matches!(
                        quantity.scalar_domain,
                        ScalarDomain::Unknown | ScalarDomain::Complex
                    )
                    || matches!(
                        quantity.physical_type.scalar,
                        ScalarDomain::Unknown | ScalarDomain::Complex
                    )
                {
                    return Err(format!(
                        "transition {:?} rational parameter conditions require known real dimensionless parameters",
                        transition.id
                    ));
                }
            }
            validate_parameter_condition_satisfiability(
                &transition.parameter_conditions,
                &transition.id,
                &quantities,
            )?;
        }

        Ok(())
    }
}

fn compare_rational_values(
    left: RationalValue,
    right: RationalValue,
) -> std::cmp::Ordering {
    let left_scaled = i128::from(left.numerator) * i128::from(right.denominator);
    let right_scaled = i128::from(right.numerator) * i128::from(left.denominator);
    left_scaled.cmp(&right_scaled)
}

fn validate_parameter_condition_satisfiability(
    conditions: &[ParameterCondition],
    transition_id: &str,
    quantities: &BTreeMap<&str, &ScientificQuantity>,
) -> Result<(), String> {
    use std::cmp::Ordering;

    let mut grouped: BTreeMap<&str, Vec<&ParameterCondition>> = BTreeMap::new();
    for condition in conditions {
        grouped
            .entry(condition.parameter_id.as_str())
            .or_default()
            .push(condition);
    }

    for (parameter_id, predicates) in grouped {
        let quantity = quantities.get(parameter_id).ok_or_else(|| {
            format!(
                "transition {transition_id:?} references undeclared parameter {parameter_id:?}"
            )
        })?;
        let integer_domain = quantity.scalar_domain == ScalarDomain::Integer;
        let mut lower: Option<(RationalValue, bool)> = None;
        let mut upper: Option<(RationalValue, bool)> = None;
        let mut equality: Option<RationalValue> = None;
        let mut exclusions = Vec::new();

        for predicate in predicates {
            let value = predicate.value;
            match predicate.relation {
                ParameterRelation::Equal => {
                    if equality.is_some_and(|current| {
                        compare_rational_values(current, value) != Ordering::Equal
                    }) {
                        return Err(format!(
                            "transition {transition_id:?} has contradictory equalities for parameter {parameter_id:?}"
                        ));
                    }
                    equality = Some(value);
                }
                ParameterRelation::NotEqual => exclusions.push(value),
                ParameterRelation::GreaterThan | ParameterRelation::GreaterOrEqual => {
                    let inclusive = predicate.relation == ParameterRelation::GreaterOrEqual;
                    match lower {
                        None => lower = Some((value, inclusive)),
                        Some((current, current_inclusive)) => {
                            match compare_rational_values(value, current) {
                                Ordering::Greater => lower = Some((value, inclusive)),
                                Ordering::Equal => {
                                    lower = Some((current, current_inclusive && inclusive));
                                }
                                Ordering::Less => {}
                            }
                        }
                    }
                }
                ParameterRelation::LessThan | ParameterRelation::LessOrEqual => {
                    let inclusive = predicate.relation == ParameterRelation::LessOrEqual;
                    match upper {
                        None => upper = Some((value, inclusive)),
                        Some((current, current_inclusive)) => {
                            match compare_rational_values(value, current) {
                                Ordering::Less => upper = Some((value, inclusive)),
                                Ordering::Equal => {
                                    upper = Some((current, current_inclusive && inclusive));
                                }
                                Ordering::Greater => {}
                            }
                        }
                    }
                }
            }
        }

        if let (Some((low, low_inclusive)), Some((high, high_inclusive))) = (lower, upper) {
            match compare_rational_values(low, high) {
                Ordering::Greater => {
                    return Err(format!(
                        "transition {transition_id:?} has an empty parameter interval for {parameter_id:?}"
                    ));
                }
                Ordering::Equal if !low_inclusive || !high_inclusive => {
                    return Err(format!(
                        "transition {transition_id:?} has an empty open parameter interval for {parameter_id:?}"
                    ));
                }
                _ => {}
            }
        }

        if let Some(value) = equality {
            if lower.is_some_and(|(low, inclusive)| {
                match compare_rational_values(value, low) {
                    Ordering::Less => true,
                    Ordering::Equal => !inclusive,
                    Ordering::Greater => false,
                }
            }) || upper.is_some_and(|(high, inclusive)| {
                match compare_rational_values(value, high) {
                    Ordering::Greater => true,
                    Ordering::Equal => !inclusive,
                    Ordering::Less => false,
                }
            }) || exclusions.iter().any(|excluded| {
                compare_rational_values(value, *excluded) == Ordering::Equal
            }) {
                return Err(format!(
                    "transition {transition_id:?} has no admissible value for parameter {parameter_id:?}"
                ));
            }
        } else if let (Some((low, low_inclusive)), Some((high, high_inclusive))) = (lower, upper)
            && compare_rational_values(low, high) == Ordering::Equal
            && exclusions.iter().any(|excluded| {
                compare_rational_values(low, *excluded) == Ordering::Equal
            })
            && low_inclusive
            && high_inclusive
        {
            return Err(format!(
                "transition {transition_id:?} excludes its only admissible value for parameter {parameter_id:?}"
            ));
        }

        if integer_domain {
            validate_integer_parameter_domain(
                parameter_id,
                transition_id,
                lower,
                upper,
                equality,
                &exclusions,
            )?;
        }
    }

    Ok(())
}

fn floor_div(numerator: i128, denominator: i128) -> i128 {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder != 0 && numerator < 0 {
        quotient - 1
    } else {
        quotient
    }
}

fn ceil_div(numerator: i128, denominator: i128) -> i128 {
    -floor_div(-numerator, denominator)
}

fn validate_integer_parameter_domain(
    parameter_id: &str,
    transition_id: &str,
    lower: Option<(RationalValue, bool)>,
    upper: Option<(RationalValue, bool)>,
    equality: Option<RationalValue>,
    exclusions: &[RationalValue],
) -> Result<(), String> {
    use std::cmp::Ordering;

    let lower_integer = lower.map(|(bound, inclusive)| {
        let numerator = i128::from(bound.numerator);
        let denominator = i128::from(bound.denominator);
        if inclusive {
            ceil_div(numerator, denominator)
        } else {
            floor_div(numerator, denominator) + 1
        }
    });
    let upper_integer = upper.map(|(bound, inclusive)| {
        let numerator = i128::from(bound.numerator);
        let denominator = i128::from(bound.denominator);
        if inclusive {
            floor_div(numerator, denominator)
        } else {
            ceil_div(numerator, denominator) - 1
        }
    });

    if let Some(value) = equality {
        let numerator = i128::from(value.numerator);
        let denominator = i128::from(value.denominator);
        if numerator % denominator != 0 {
            return Err(format!(
                "transition {transition_id:?} equality for integer parameter {parameter_id:?} is non-integral"
            ));
        }
        let integer = numerator / denominator;
        if lower_integer.is_some_and(|minimum| integer < minimum)
            || upper_integer.is_some_and(|maximum| integer > maximum)
            || exclusions.iter().any(|excluded| {
                compare_rational_values(
                    value,
                    *excluded,
                ) == Ordering::Equal
            })
        {
            return Err(format!(
                "transition {transition_id:?} has no admissible integer value for parameter {parameter_id:?}"
            ));
        }
        return Ok(());
    }

    if let (Some(minimum), Some(maximum)) = (lower_integer, upper_integer) {
        if minimum > maximum {
            return Err(format!(
                "transition {transition_id:?} has no admissible integer value for parameter {parameter_id:?}"
            ));
        }
        let count = maximum - minimum + 1;
        let excluded_integers = exclusions
            .iter()
            .filter_map(|value| {
                let numerator = i128::from(value.numerator);
                let denominator = i128::from(value.denominator);
                (numerator % denominator == 0).then_some(numerator / denominator)
            })
            .filter(|value| *value >= minimum && *value <= maximum)
            .collect::<BTreeSet<_>>();
        if (excluded_integers.len() as i128) >= count {
            return Err(format!(
                "transition {transition_id:?} excludes every integer value for parameter {parameter_id:?}"
            ));
        }
    }

    Ok(())
}

fn ensure_unique_ids<'a>(
    values: impl IntoIterator<Item = &'a str>,
    label: &str,
) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_identifier(value)?;
        if !seen.insert(value) {
            return Err(format!("duplicate hybrid {label} ID {value:?}"));
        }
    }
    Ok(())
}

fn validate_references(
    ids: &[String],
    available: &BTreeSet<&str>,
    label: &str,
) -> Result<(), String> {
    ensure_unique_ids(ids.iter().map(String::as_str), label)?;
    for id in ids {
        if !available.contains(id.as_str()) {
            return Err(format!("unresolved {label} reference {id:?}"));
        }
    }
    Ok(())
}

fn validate_parameters(
    ids: &[String],
    quantities: &BTreeMap<&str, &ScientificQuantity>,
) -> Result<(), String> {
    ensure_unique_ids(ids.iter().map(String::as_str), "parameter reference")?;
    for id in ids {
        let quantity = quantities
            .get(id.as_str())
            .ok_or_else(|| format!("unresolved hybrid parameter reference {id:?}"))?;
        if quantity.role != QuantityRole::Parameter {
            return Err(format!("hybrid parameter reference {id:?} is not a parameter"));
        }
    }
    Ok(())
}

fn validate_referenced_parameters(
    expression: &ScientificExpression,
    quantities: &BTreeMap<&str, &ScientificQuantity>,
    declared_parameter_ids: &[String],
    label: &str,
    record_id: &str,
) -> Result<(), String> {
    let declared: BTreeSet<&str> = declared_parameter_ids.iter().map(String::as_str).collect();
    let referenced = expression_symbols(expression);
    for symbol in referenced {
        if quantities
            .get(symbol.as_str())
            .is_some_and(|quantity| quantity.role == QuantityRole::Parameter)
            && !declared.contains(symbol.as_str())
        {
            return Err(format!(
                "{label} {record_id:?} uses parameter {symbol:?} outside its declared parameter roster"
            ));
        }
    }
    Ok(())
}

fn validate_time_dependence(
    expression: &ScientificExpression,
    time_symbol: Option<&str>,
    quantities: &BTreeMap<&str, &ScientificQuantity>,
    label: &str,
    record_id: &str,
) -> Result<(), String> {
    let referenced = explicit_expression_symbols(expression);
    let time_symbols: Vec<&str> = quantities
        .iter()
        .filter_map(|(name, quantity)| {
            (quantity.role == QuantityRole::Coordinate
                && quantity.physical_type.dimension == Some(PhysicalDimension::TIME))
            .then_some(*name)
        })
        .collect();
    let mentions_time = time_symbols
        .iter()
        .any(|time_symbol| referenced.contains(*time_symbol));
    match time_symbol {
        Some(symbol) => {
            let quantity = quantities.get(symbol).ok_or_else(|| {
                format!("{label} {record_id:?} references undeclared time symbol {symbol:?}")
            })?;
            if quantity.role != QuantityRole::Coordinate
                || quantity.physical_type.dimension != Some(PhysicalDimension::TIME)
            {
                return Err(format!(
                    "{label} {record_id:?} time symbol {symbol:?} is not a time coordinate"
                ));
            }
            if !referenced.contains(symbol) {
                return Err(format!(
                    "{label} {record_id:?} declares unused time dependence {symbol:?}"
                ));
            }
            if time_symbols.iter().any(|other| {
                *other != symbol && referenced.contains(*other)
            }) {
                return Err(format!(
                    "{label} {record_id:?} uses a time coordinate other than the declared hybrid clock"
                ));
            }
        }
        None if mentions_time => {
            return Err(format!(
                "{label} {record_id:?} uses time without declaring its time symbol"
            ));
        }
        None => {}
    }
    Ok(())
}

fn explicit_expression_symbols(expression: &ScientificExpression) -> BTreeSet<String> {
    fn visit(expression: &ScientificExpression, symbols: &mut BTreeSet<String>) {
        match expression {
            ScientificExpression::Symbol(name) => {
                symbols.insert(name.clone());
            }
            ScientificExpression::Rational { .. } => {}
            ScientificExpression::Add(terms) | ScientificExpression::Multiply(terms) => {
                for term in terms {
                    visit(term, symbols);
                }
            }
            ScientificExpression::Negate(inner) => visit(inner, symbols),
            ScientificExpression::Derivative { expression, .. } => {
                // The independent variable defines the derivative; it is not
                // an explicit time argument of the resulting guard/reset value.
                visit(expression, symbols);
            }
        }
    }
    let mut symbols = BTreeSet::new();
    visit(expression, &mut symbols);
    symbols
}

fn expression_derivative_variables(expression: &ScientificExpression) -> BTreeSet<String> {
    fn visit(expression: &ScientificExpression, output: &mut BTreeSet<String>) {
        match expression {
            ScientificExpression::Symbol(_) | ScientificExpression::Rational { .. } => {}
            ScientificExpression::Add(terms) | ScientificExpression::Multiply(terms) => {
                for term in terms {
                    visit(term, output);
                }
            }
            ScientificExpression::Negate(inner) => visit(inner, output),
            ScientificExpression::Derivative {
                expression,
                with_respect_to,
                ..
            } => {
                output.insert(with_respect_to.clone());
                visit(expression, output);
            }
        }
    }
    let mut output = BTreeSet::new();
    visit(expression, &mut output);
    output
}

fn expression_symbols(expression: &ScientificExpression) -> BTreeSet<String> {
    fn visit(expression: &ScientificExpression, symbols: &mut BTreeSet<String>) {
        match expression {
            ScientificExpression::Symbol(name) => {
                symbols.insert(name.clone());
            }
            ScientificExpression::Rational { .. } => {}
            ScientificExpression::Add(terms) | ScientificExpression::Multiply(terms) => {
                for term in terms {
                    visit(term, symbols);
                }
            }
            ScientificExpression::Negate(inner) => visit(inner, symbols),
            ScientificExpression::Derivative {
                expression,
                with_respect_to,
                ..
            } => {
                visit(expression, symbols);
                symbols.insert(with_respect_to.clone());
            }
        }
    }
    let mut symbols = BTreeSet::new();
    visit(expression, &mut symbols);
    symbols
}

fn infer_expression_type(
    expression: &ScientificExpression,
    quantities: &BTreeMap<&str, &ScientificQuantity>,
) -> Result<Option<PhysicalType>, String> {
    match expression {
        ScientificExpression::Symbol(name) => {
            let quantity = quantities
                .get(name.as_str())
                .ok_or_else(|| format!("expression references undeclared symbol {name:?}"))?;
            if quantity.physical_type.dimension.is_none()
                || quantity.physical_type.scalar == ScalarDomain::Unknown
            {
                return Err(format!(
                    "expression symbol {name:?} has unknown physical dimension or scalar domain"
                ));
            }
            Ok(Some(quantity.physical_type.clone()))
        }
        ScientificExpression::Rational {
            numerator,
            denominator,
        } => {
            if *denominator == 0 {
                return Err("expression rational denominator must be non-zero".into());
            }
            if *numerator == 0 {
                Ok(None)
            } else {
                Ok(Some(PhysicalType::dimensionless()))
            }
        }
        ScientificExpression::Add(terms) => {
            if terms.is_empty() {
                return Err("hybrid expression addition cannot be empty".into());
            }
            let mut result: Option<PhysicalType> = None;
            for term in terms {
                if let Some(term_type) = infer_expression_type(term, quantities)? {
                    if let Some(current) = &result {
                        match current.judge_compatibility(&term_type) {
                            TypeJudgement::Valid(()) => {}
                            TypeJudgement::Invalid(error) => return Err(error.reason),
                            TypeJudgement::Unknown(reason) => return Err(reason),
                        }
                    } else {
                        result = Some(term_type);
                    }
                }
            }
            Ok(result)
        }
        ScientificExpression::Multiply(factors) => {
            if factors.is_empty() {
                return Err("hybrid expression multiplication cannot be empty".into());
            }
            let mut dimension = PhysicalDimension::DIMENSIONLESS;
            let mut scalar = ScalarDomain::Real;
            for factor in factors {
                let Some(factor_type) = infer_expression_type(factor, quantities)? else {
                    // Zero is a polymorphic additive/multiplicative annihilator;
                    // the surrounding equation supplies its contextual type.
                    return Ok(None);
                };
                let factor_dimension = factor_type.dimension.ok_or_else(|| {
                    "hybrid multiplication requires known factor dimensions".to_string()
                })?;
                dimension = dimension.checked_add(factor_dimension).ok_or_else(|| {
                    "hybrid multiplication physical-dimension exponent overflow".to_string()
                })?;
                scalar = match (scalar, factor_type.scalar) {
                    (ScalarDomain::Unknown, _) | (_, ScalarDomain::Unknown) => {
                        ScalarDomain::Unknown
                    }
                    (ScalarDomain::Complex, _) | (_, ScalarDomain::Complex) => {
                        ScalarDomain::Complex
                    }
                    (ScalarDomain::ApproximateReal, _)
                    | (_, ScalarDomain::ApproximateReal) => ScalarDomain::ApproximateReal,
                    _ => ScalarDomain::Real,
                };
            }
            if scalar == ScalarDomain::Unknown {
                return Err("hybrid multiplication has an unknown scalar domain".into());
            }
            let mut result = PhysicalType::with_kind(QuantityKind::Unknown, dimension);
            result.scalar = scalar;
            Ok(Some(result))
        }
        ScientificExpression::Negate(inner) => infer_expression_type(inner, quantities),
        ScientificExpression::Derivative {
            expression,
            with_respect_to,
            order,
        } => {
            if *order == 0 {
                return Err("hybrid derivative order must be greater than zero".into());
            }
            let Some(value_type) = infer_expression_type(expression, quantities)? else {
                return Ok(None);
            };
            let independent = quantities
                .get(with_respect_to.as_str())
                .ok_or_else(|| {
                    format!("derivative references undeclared symbol {with_respect_to:?}")
                })?;
            let mut result = value_type;
            for _ in 0..*order {
                result = match result.derivative_with(&independent.physical_type) {
                    TypeJudgement::Valid(value) => value,
                    TypeJudgement::Invalid(error) => return Err(error.reason),
                    TypeJudgement::Unknown(reason) => return Err(reason),
                };
            }
            Ok(Some(result))
        }
    }
}

fn check_expression_type(
    expression: &ScientificExpression,
    expected: &PhysicalType,
    quantities: &BTreeMap<&str, &ScientificQuantity>,
) -> Result<(), String> {
    let Some(actual) = infer_expression_type(expression, quantities)? else {
        // Typed contextual zero is valid for reset values and zero vector-field
        // components (e.g. v_dot = 0), but unknown nonzero expressions fail.
        return Ok(());
    };
    match expected.judge_compatibility(&actual) {
        TypeJudgement::Valid(()) => Ok(()),
        TypeJudgement::Invalid(error) => Err(error.reason),
        TypeJudgement::Unknown(reason) => Err(reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific_model::{
        EvidenceClass, ModelAssumption, ModelEvidenceRequirement, ModelNonClaim, SolverIntent,
    };
    use crate::ModelMaturity;

    fn quantity(
        name: &str,
        kind: QuantityKind,
        dimension: PhysicalDimension,
        role: QuantityRole,
    ) -> ScientificQuantity {
        ScientificQuantity {
            name: name.into(),
            physical_type: PhysicalType::with_kind(kind, dimension),
            scalar_domain: ScalarDomain::Real,
            role,
        }
    }

    fn sym(name: &str) -> ScientificExpression {
        ScientificExpression::symbol(name).expect("valid symbol")
    }

    fn derivative(name: &str, time: &str) -> ScientificExpression {
        ScientificExpression::Derivative {
            expression: Box::new(sym(name)),
            with_respect_to: time.into(),
            order: 1,
        }
    }

    fn point_impact_model() -> ScientificModel {
        let quantities = vec![
            quantity("y", QuantityKind::Length, PhysicalDimension::LENGTH, QuantityRole::State),
            quantity(
                "v",
                QuantityKind::Velocity,
                PhysicalDimension::VELOCITY,
                QuantityRole::State,
            ),
            quantity(
                "g",
                QuantityKind::Acceleration,
                PhysicalDimension::ACCELERATION,
                QuantityRole::Parameter,
            ),
            quantity("m", QuantityKind::Mass, PhysicalDimension::MASS, QuantityRole::Parameter),
            quantity(
                "e",
                QuantityKind::Dimensionless,
                PhysicalDimension::DIMENSIONLESS,
                QuantityRole::Parameter,
            ),
            quantity(
                "theta",
                QuantityKind::Dimensionless,
                PhysicalDimension::DIMENSIONLESS,
                QuantityRole::Parameter,
            ),
            quantity("N", QuantityKind::Force, PhysicalDimension::FORCE, QuantityRole::Auxiliary),
            quantity("t", QuantityKind::Time, PhysicalDimension::TIME, QuantityRole::Coordinate),
        ];
        ScientificModel::new(
            "point-impact-hybrid-v0-2",
            "hybrid-mechanics",
            ModelMaturity::ResearchPrototype,
            quantities,
            // This kinematic identity holds in both free flight and persistent
            // contact. Mode-specific acceleration/reaction equations live only
            // in HybridMode.equations below.
            vec![ScientificEquation {
                name: "kinematic-y-dot".into(),
                left: derivative("y", "t"),
                right: sym("v"),
            }],
            vec![
                ModelAssumption {
                    id: "constant-gravity".into(),
                    statement: "Gravity magnitude is positive and constant.".into(),
                },
                ModelAssumption {
                    id: "restitution-domain".into(),
                    statement: "Newton restitution coefficient lies in [0, 1].".into(),
                },
            ],
            SolverIntent {
                backend_family: None,
                conservation_required: true,
                positivity_required: true,
                invariant_domain_required: true,
                monotonicity_required: false,
                target_accuracy_order: Some(1),
            },
            vec![
                ModelEvidenceRequirement {
                    id: "analytic-impact".into(),
                    class: EvidenceClass::Formal,
                    statement: "The guard crossing and reset are checked against the analytic point-impact solution.".into(),
                },
                ModelEvidenceRequirement {
                    id: "energy-ledger".into(),
                    class: EvidenceClass::Numerical,
                    statement: "Pre/post-impact kinetic energy and physical impact dissipation are separately reconciled.".into(),
                },
            ],
            vec![ModelNonClaim {
                id: "not-empirical-validation".into(),
                statement: "A synthetic analytic fixture is not physical experimental validation.".into(),
            }],
            vec!["fixture:point-impact-hybrid-v0-2".into()],
        )
        .unwrap()
    }

    fn hybrid_fixture() -> HybridDynamicsV0_1 {
        let modes = vec![
            HybridMode {
                id: "free-flight".into(),
                equations: vec![
                    ScientificEquation {
                        name: "y-dot".into(),
                        left: derivative("y", "t"),
                        right: sym("v"),
                    },
                    ScientificEquation {
                        name: "v-dot".into(),
                        left: derivative("v", "t"),
                        right: ScientificExpression::Negate(Box::new(sym("g"))),
                    },
                ],
                parameter_ids: vec!["g".into()],
                validity_assumption_ids: vec!["constant-gravity".into()],
                evidence_requirement_ids: vec!["analytic-impact".into()],
                non_claim_ids: vec!["not-empirical-validation".into()],
            },
            HybridMode {
                id: "persistent-contact".into(),
                equations: vec![
                    ScientificEquation {
                        name: "contact-position-constraint".into(),
                        left: sym("y"),
                        right: ScientificExpression::rational(0, 1).unwrap(),
                    },
                    ScientificEquation {
                        name: "contact-velocity-constraint".into(),
                        left: sym("v"),
                        right: ScientificExpression::rational(0, 1).unwrap(),
                    },
                    ScientificEquation {
                        name: "y-dot".into(),
                        left: derivative("y", "t"),
                        right: ScientificExpression::rational(0, 1).unwrap(),
                    },
                    ScientificEquation {
                        name: "v-dot".into(),
                        left: derivative("v", "t"),
                        right: ScientificExpression::rational(0, 1).unwrap(),
                    },
                    ScientificEquation {
                        name: "normal-reaction".into(),
                        left: sym("N"),
                        right: ScientificExpression::Multiply(vec![sym("m"), sym("g")]),
                    },
                ],
                parameter_ids: vec!["g".into(), "m".into()],
                validity_assumption_ids: vec!["constant-gravity".into()],
                evidence_requirement_ids: vec!["energy-ledger".into()],
                non_claim_ids: vec!["not-empirical-validation".into()],
            },
        ];
        let guards = vec![GuardSurface {
            id: "ground-impact".into(),
            source_mode_id: "free-flight".into(),
            expression: sym("y"),
            crossing_direction: GuardCrossingDirection::PositiveToNegative,
            time_symbol: None,
            parameter_ids: vec![],
            validity_assumption_ids: vec!["constant-gravity".into()],
            evidence_requirement_ids: vec!["analytic-impact".into()],
        }];
        let resets = vec![
            ResetMap {
                id: "rebound-reset".into(),
                source_mode_id: "free-flight".into(),
                target_mode_id: "free-flight".into(),
                assignments: vec![
                    ResetAssignment {
                        target_state_id: "y".into(),
                        expression: ScientificExpression::rational(0, 1).unwrap(),
                    },
                    ResetAssignment {
                        target_state_id: "v".into(),
                        expression: ScientificExpression::Multiply(vec![
                            ScientificExpression::rational(-1, 1).unwrap(),
                            sym("e"),
                            sym("v"),
                        ]),
                    },
                ],
                law: ResetLaw::NewtonRestitution {
                    restitution_parameter_id: "e".into(),
                },
                time_symbol: None,
                parameter_ids: vec!["e".into()],
                validity_assumption_ids: vec!["restitution-domain".into()],
                evidence_requirement_ids: vec!["analytic-impact".into()],
            },
            ResetMap {
                id: "settle-reset".into(),
                source_mode_id: "free-flight".into(),
                target_mode_id: "persistent-contact".into(),
                assignments: vec![
                    ResetAssignment {
                        target_state_id: "y".into(),
                        expression: ScientificExpression::rational(0, 1).unwrap(),
                    },
                    ResetAssignment {
                        target_state_id: "v".into(),
                        expression: ScientificExpression::rational(0, 1).unwrap(),
                    },
                ],
                law: ResetLaw::PersistentContact {
                    law_id: "unilateral-normal-support-v0".into(),
                },
                time_symbol: None,
                parameter_ids: vec![],
                validity_assumption_ids: vec!["constant-gravity".into()],
                evidence_requirement_ids: vec!["energy-ledger".into()],
            },
        ];
        let transitions = vec![
            HybridTransition {
                id: "rebound".into(),
                source_mode_id: "free-flight".into(),
                guard_id: "ground-impact".into(),
                reset_id: "rebound-reset".into(),
                target_mode_id: "free-flight".into(),
                parameter_conditions: vec![
                    ParameterCondition {
                        parameter_id: "e".into(),
                        relation: ParameterRelation::GreaterThan,
                        value: RationalValue::new(0, 1).unwrap(),
                    },
                    ParameterCondition {
                        parameter_id: "e".into(),
                        relation: ParameterRelation::LessOrEqual,
                        value: RationalValue::new(1, 1).unwrap(),
                    },
                ],
                validity_assumption_ids: vec!["restitution-domain".into()],
                evidence_requirement_ids: vec!["analytic-impact".into()],
            },
            HybridTransition {
                id: "settle".into(),
                source_mode_id: "free-flight".into(),
                guard_id: "ground-impact".into(),
                reset_id: "settle-reset".into(),
                target_mode_id: "persistent-contact".into(),
                parameter_conditions: vec![ParameterCondition {
                    parameter_id: "e".into(),
                    relation: ParameterRelation::Equal,
                    value: RationalValue::new(0, 1).unwrap(),
                }],
                validity_assumption_ids: vec!["constant-gravity".into()],
                evidence_requirement_ids: vec!["energy-ledger".into()],
            },
        ];
        HybridDynamicsV0_1::new(
            "t",
            "free-flight",
            modes,
            guards,
            resets,
            transitions,
            SimultaneousEventPolicy::RequireJointEventCertificate,
        )
        .unwrap()
    }

    #[test]
    fn hybrid_point_impact_fixture_is_valid_and_canonical() {
        let model = point_impact_model()
            .with_hybrid_dynamics(hybrid_fixture())
            .expect("hybrid model is valid");
        assert_eq!(
            model.hybrid_dynamics.as_ref().unwrap().schema_revision,
            HYBRID_DYNAMICS_SCHEMA_V0_1
        );
        assert_eq!(
            model.hybrid_dynamics.as_ref().unwrap().modes[0].id,
            "free-flight"
        );
    }

    #[test]
    fn hybrid_extension_changes_parent_model_digest() {
        let base = point_impact_model();
        let hybrid = base.clone()
            .with_hybrid_dynamics(hybrid_fixture())
            .expect("valid hybrid extension");

        assert_ne!(base.digest(), hybrid.digest());
    }

    #[test]
    fn reset_must_assign_every_state_coordinate() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.resets[0].assignments.pop();

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn guard_expression_must_be_dimensionally_typed() {
        let mut model = point_impact_model();
        let restitution_index = model
            .quantities
            .iter()
            .position(|quantity| quantity.name == "e")
            .unwrap();
        model.quantities[restitution_index].physical_type = PhysicalType::unknown();
        model.quantities[restitution_index].scalar_domain = ScalarDomain::Unknown;

        let mut hybrid = hybrid_fixture();
        hybrid.guards[0].expression = sym("e");
        hybrid.guards[0].parameter_ids = vec!["e".into()];

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn reset_cannot_assign_velocity_to_position() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.resets[0].assignments[0].expression = sym("v");

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn e_zero_persistent_contact_is_a_distinct_transition_and_mode() {
        let model = point_impact_model()
            .with_hybrid_dynamics(hybrid_fixture())
            .expect("valid hybrid model");
        let hybrid = model.hybrid_dynamics.unwrap();
        let settle = hybrid.transitions.iter().find(|item| item.id == "settle").unwrap();
        let reset = hybrid.resets.iter().find(|item| item.id == settle.reset_id).unwrap();

        assert_eq!(reset.target_mode_id, "persistent-contact");
        assert_eq!(reset.law, ResetLaw::PersistentContact {
            law_id: "unilateral-normal-support-v0".into()
        });
    }

    #[test]
    fn parameter_and_mode_references_fail_closed() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        let settle = hybrid
            .transitions
            .iter_mut()
            .find(|transition| transition.id == "settle")
            .unwrap();
        settle.parameter_conditions[0].parameter_id = "missing".into();

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn time_dependence_must_name_a_time_coordinate() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.guards[0].expression = ScientificExpression::Add(vec![
            sym("y"),
            ScientificExpression::Multiply(vec![sym("v"), sym("t")]),
        ]);

        assert!(hybrid.validate(&model).is_err());
        hybrid.guards[0].time_symbol = Some("t".into());
        // The guard is now time-declared, but its length terms must remain
        // dimensionally compatible. y + v*t is valid.
        assert!(hybrid.validate(&model).is_ok());
    }
    #[test]
    fn legacy_models_remain_v0_1_without_hybrid_extension() {
        let model = point_impact_model();
        assert_eq!(
            model.schema_revision,
            crate::scientific_model::SCIENTIFIC_MODEL_SCHEMA_V0_1
        );
        assert!(model.hybrid_dynamics.is_none());
        assert!(model.validate().is_ok());
    }

    #[test]
    fn hybrid_models_are_promoted_to_v0_2_and_bind_hybrid_semantics() {
        let model = point_impact_model()
            .with_hybrid_dynamics(hybrid_fixture())
            .expect("valid hybrid model");
        assert_eq!(
            model.schema_revision,
            crate::scientific_model::SCIENTIFIC_MODEL_SCHEMA_V0_2
        );
        assert!(model.hybrid_dynamics.is_some());
        assert!(model.validate().is_ok());
    }

    #[test]
    fn mode_parameter_roster_must_include_every_used_parameter() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.modes
            .iter_mut()
            .find(|mode| mode.id == "free-flight")
            .unwrap()
            .parameter_ids
            .clear();

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn reset_parameter_roster_must_include_every_used_parameter() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.resets
            .iter_mut()
            .find(|reset| reset.id == "rebound-reset")
            .unwrap()
            .parameter_ids
            .clear();

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn time_dependent_guard_must_use_the_global_hybrid_clock() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.guards[0].expression = ScientificExpression::Add(vec![
            sym("y"),
            ScientificExpression::Multiply(vec![sym("v"), sym("t")]),
        ]);
        hybrid.guards[0].time_symbol = Some("t".into());
        assert!(hybrid.validate(&model).is_ok());

        hybrid.guards[0].time_symbol = Some("not-the-clock".into());
        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn mode_equation_derivative_must_use_the_global_hybrid_clock() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.modes[0].equations[0].left = derivative("y", "v");

        assert!(hybrid.validate(&model).is_err());
    }

    #[test]
    fn rational_conditions_require_canonical_reduced_values() {
        assert_eq!(
            RationalValue::new(2, 2).unwrap(),
            RationalValue {
                numerator: 1,
                denominator: 1
            }
        );
        let mut hybrid = hybrid_fixture();
        let settle = hybrid
            .transitions
            .iter_mut()
            .find(|transition| transition.id == "settle")
            .unwrap();
        settle.parameter_conditions[0].value = RationalValue {
            numerator: 2,
            denominator: 2
        };

        assert!(hybrid.validate(&point_impact_model()).is_err());
    }

    #[test]
    fn transition_parameter_conditions_require_known_real_dimensionless_parameters() {
        let mut model = point_impact_model();
        let index = model
            .quantities
            .iter()
            .position(|quantity| quantity.name == "theta")
            .unwrap();
        model.quantities[index].scalar_domain = ScalarDomain::Unknown;
        model.quantities[index].physical_type.scalar = ScalarDomain::Unknown;

        let mut hybrid = hybrid_fixture();
        let settle = hybrid
            .transitions
            .iter_mut()
            .find(|transition| transition.id == "settle")
            .unwrap();
        settle.parameter_conditions = vec![ParameterCondition {
            parameter_id: "theta".into(),
            relation: ParameterRelation::Equal,
            value: RationalValue::new(0, 1).unwrap(),
        }];

        assert!(hybrid.validate(&model).is_err());
    }


    #[test]
    fn legacy_v0_1_serialization_omits_optional_hybrid_field() {
        let model = point_impact_model();
        let value = serde_json::to_value(&model).expect("serialize legacy model");
        assert!(value.get("hybrid_dynamics").is_none());

        let round_trip: ScientificModel =
            serde_json::from_value(value).expect("deserialize legacy v0.1 model");
        assert_eq!(round_trip, model);
        assert_eq!(
            round_trip.schema_revision,
            crate::scientific_model::SCIENTIFIC_MODEL_SCHEMA_V0_1
        );
    }

    #[test]
    fn v0_2_serialization_includes_hybrid_semantics_in_parent_identity() {
        let model = point_impact_model()
            .with_hybrid_dynamics(hybrid_fixture())
            .expect("valid hybrid model");
        let value = serde_json::to_value(&model).expect("serialize hybrid model");
        assert!(value.get("hybrid_dynamics").is_some());

        let round_trip: ScientificModel =
            serde_json::from_value(value).expect("deserialize hybrid model");
        assert_eq!(round_trip, model);
        assert_eq!(round_trip.digest(), model.digest());
    }

    #[test]
    fn canonical_hybrid_construction_is_independent_of_input_order() {
        let forward = hybrid_fixture();
        let mut reversed = hybrid_fixture();
        reversed.modes.reverse();
        reversed.guards.reverse();
        reversed.resets.reverse();
        reversed.transitions.reverse();

        // Both fixtures use unique IDs and the same semantic contents.
        let left = point_impact_model()
            .with_hybrid_dynamics(forward)
            .expect("forward model");
        let right = point_impact_model()
            .with_hybrid_dynamics(reversed)
            .expect("reversed model");

        assert_eq!(left.digest(), right.digest());
    }


    #[test]
    fn time_dependent_guard_rejects_a_second_time_coordinate() {
        let mut model = point_impact_model();
        model.quantities.push(quantity(
            "t2",
            QuantityKind::Time,
            PhysicalDimension::TIME,
            QuantityRole::Coordinate,
        ));
        let mut hybrid = hybrid_fixture();
        hybrid.guards[0].expression = ScientificExpression::Add(vec![
            sym("y"),
            ScientificExpression::Multiply(vec![sym("v"), sym("t2")]),
        ]);
        hybrid.guards[0].time_symbol = Some("t".into());

        assert!(hybrid.validate(&model).is_err());
    }


    #[test]
    fn schema_v0_2_rejects_missing_hybrid_payload() {
        let mut model = point_impact_model();
        model.schema_revision = crate::scientific_model::SCIENTIFIC_MODEL_SCHEMA_V0_2.into();

        assert!(model.validate().is_err());
    }

    #[test]
    fn both_impact_resets_project_position_onto_the_guard() {
        let hybrid = hybrid_fixture();
        for reset in &hybrid.resets {
            let y_assignment = reset
                .assignments
                .iter()
                .find(|assignment| assignment.target_state_id == "y")
                .expect("every reset assigns position");
            assert_eq!(
                y_assignment.expression,
                ScientificExpression::rational(0, 1).unwrap()
            );
        }
    }


    #[test]
    fn newton_restitution_rejects_complex_coefficients() {
        let mut model = point_impact_model();
        let index = model
            .quantities
            .iter()
            .position(|quantity| quantity.name == "e")
            .unwrap();
        model.quantities[index].scalar_domain = ScalarDomain::Complex;
        model.quantities[index].physical_type.scalar = ScalarDomain::Complex;

        assert!(hybrid_fixture().validate(&model).is_err());
    }


    #[test]
    fn canonical_digest_normalizes_mutated_hybrid_roster_order() {
        let model = point_impact_model()
            .with_hybrid_dynamics(hybrid_fixture())
            .expect("valid canonical model");
        let mut permuted = model.clone();
        let hybrid = permuted.hybrid_dynamics.as_mut().unwrap();

        hybrid.modes.reverse();
        hybrid.guards.reverse();
        hybrid.resets.reverse();
        hybrid.transitions.reverse();
        for mode in &mut hybrid.modes {
            mode.parameter_ids.reverse();
            mode.validity_assumption_ids.reverse();
            mode.evidence_requirement_ids.reverse();
            mode.non_claim_ids.reverse();
            mode.equations.reverse();
        }
        for guard in &mut hybrid.guards {
            guard.parameter_ids.reverse();
            guard.validity_assumption_ids.reverse();
            guard.evidence_requirement_ids.reverse();
        }
        for reset in &mut hybrid.resets {
            reset.assignments.reverse();
            reset.parameter_ids.reverse();
            reset.validity_assumption_ids.reverse();
            reset.evidence_requirement_ids.reverse();
        }
        for transition in &mut hybrid.transitions {
            transition.parameter_conditions.reverse();
            transition.validity_assumption_ids.reverse();
            transition.evidence_requirement_ids.reverse();
        }

        assert_eq!(model.digest(), permuted.digest());
    }


    #[test]
    fn persistent_contact_mode_declares_unilateral_position_and_velocity_constraints() {
        let hybrid = hybrid_fixture();
        let mode = hybrid
            .modes
            .iter()
            .find(|mode| mode.id == "persistent-contact")
            .expect("persistent-contact mode");
        let names: BTreeSet<&str> = mode.equations.iter().map(|equation| equation.name.as_str()).collect();

        assert!(names.contains("contact-position-constraint"));
        assert!(names.contains("contact-velocity-constraint"));
        assert!(names.contains("y-dot"));
        assert!(names.contains("v-dot"));
        assert!(names.contains("normal-reaction"));
    }


    #[test]
    fn newton_restitution_transition_requires_strict_positive_and_unit_upper_bounds() {
        let model = point_impact_model();

        let mut missing_lower = hybrid_fixture();
        missing_lower.transitions[0]
            .parameter_conditions
            .retain(|condition| condition.relation != ParameterRelation::GreaterThan);
        assert!(missing_lower.validate(&model).is_err());

        let mut missing_upper = hybrid_fixture();
        missing_upper.transitions[0]
            .parameter_conditions
            .retain(|condition| condition.relation != ParameterRelation::LessOrEqual);
        assert!(missing_upper.validate(&model).is_err());

        let mut includes_zero = hybrid_fixture();
        let lower = includes_zero.transitions[0]
            .parameter_conditions
            .iter_mut()
            .find(|condition| condition.relation == ParameterRelation::GreaterThan)
            .unwrap();
        lower.relation = ParameterRelation::GreaterOrEqual;
        assert!(includes_zero.validate(&model).is_err());

        // The e=0 branch is still valid because it has a distinct persistent
        // contact reset and does not use NewtonRestitution.
        assert!(hybrid_fixture().validate(&model).is_ok());
    }


    #[test]
    fn derivative_coordinate_is_not_mistaken_for_explicit_guard_time_dependence() {
        let model = point_impact_model();
        let mut hybrid = hybrid_fixture();
        hybrid.guards[0].expression = derivative("y", "t");
        hybrid.guards[0].time_symbol = None;

        assert!(hybrid.validate(&model).is_ok());
    }


    #[test]
    fn transition_parameter_predicates_must_define_a_nonempty_exact_interval() {
        let model = point_impact_model();

        let mut contradictory_bounds = hybrid_fixture();
        contradictory_bounds.transitions[0].parameter_conditions.push(ParameterCondition {
            parameter_id: "e".into(),
            relation: ParameterRelation::LessThan,
            value: RationalValue::new(0, 1).unwrap(),
        });
        assert!(contradictory_bounds.validate(&model).is_err());

        let mut excluded_singleton = hybrid_fixture();
        excluded_singleton.transitions[1].parameter_conditions.push(ParameterCondition {
            parameter_id: "e".into(),
            relation: ParameterRelation::NotEqual,
            value: RationalValue::new(0, 1).unwrap(),
        });
        assert!(excluded_singleton.validate(&model).is_err());

        assert!(hybrid_fixture().validate(&model).is_ok());
    }


    #[test]
    fn integer_parameter_intervals_must_contain_an_admissible_integer() {
        let mut model = point_impact_model();
        let theta_index = model
            .quantities
            .iter()
            .position(|quantity| quantity.name == "theta")
            .unwrap();
        model.quantities[theta_index].scalar_domain = ScalarDomain::Integer;
        model.quantities[theta_index].physical_type.scalar = ScalarDomain::Integer;

        let mut hybrid = hybrid_fixture();
        let settle = hybrid
            .transitions
            .iter_mut()
            .find(|transition| transition.id == "settle")
            .unwrap();
        settle.parameter_conditions = vec![
            ParameterCondition {
                parameter_id: "theta".into(),
                relation: ParameterRelation::GreaterThan,
                value: RationalValue::new(0, 1).unwrap(),
            },
            ParameterCondition {
                parameter_id: "theta".into(),
                relation: ParameterRelation::LessThan,
                value: RationalValue::new(1, 1).unwrap(),
            },
        ];

        assert!(hybrid.validate(&model).is_err());
    }


}
