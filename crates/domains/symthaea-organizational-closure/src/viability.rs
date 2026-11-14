//! Bounded finite surrogate for viability-localized organizational witnesses.
//!
//! This module does not implement an elementary topos, a Lawvere–Tierney
//! topology, or the full categorical AP1–AP4 formulation. It checks explicit
//! finite data: viability-equivalence classes, realization and boundary maps,
//! admissibility, and finite local constraints. All conclusions are relative
//! to the supplied frozen model and its encoding.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

string_id!(ProductionStateId);
string_id!(OrganizationStateId);
string_id!(BoundaryStateId);
string_id!(ViabilityClassId);
string_id!(VariableId);
string_id!(ValueId);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteState {
    pub id: String,
    pub viability_class: ViabilityClassId,
    pub viable: bool,
}

impl FiniteState {
    pub fn new(
        id: impl Into<String>,
        viability_class: impl Into<ViabilityClassId>,
        viable: bool,
    ) -> Self {
        Self {
            id: id.into(),
            viability_class: viability_class.into(),
            viable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalConstraint {
    pub id: String,
    pub scope: Vec<VariableId>,
    /// Each tuple is an allowed joint assignment in the same order as scope.
    pub allowed_assignments: BTreeSet<Vec<ValueId>>,
}

impl LocalConstraint {
    pub fn new(
        id: impl Into<String>,
        scope: Vec<VariableId>,
        allowed_assignments: impl IntoIterator<Item = Vec<ValueId>>,
    ) -> Self {
        Self {
            id: id.into(),
            scope,
            allowed_assignments: allowed_assignments.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiniteViabilityModel {
    pub production_states: BTreeMap<ProductionStateId, FiniteState>,
    pub organization_states: BTreeMap<OrganizationStateId, FiniteState>,
    pub boundary_states: BTreeMap<BoundaryStateId, FiniteState>,
    /// Realization map from production states to organization states.
    pub realization: BTreeMap<ProductionStateId, OrganizationStateId>,
    /// Boundary-production map from production states to boundary states.
    pub boundary_production: BTreeMap<ProductionStateId, BoundaryStateId>,
    pub required_boundary_classes: BTreeSet<ViabilityClassId>,
    /// Frozen admissibility predicate represented extensionally as allowed IDs.
    pub admissible_boundaries: BTreeSet<BoundaryStateId>,
    /// Finite variables and their non-empty domains.
    pub variables: BTreeMap<VariableId, BTreeSet<ValueId>>,
    pub constraints: Vec<LocalConstraint>,
    /// Counts expanded search nodes, not elapsed time or wall-clock operations.
    pub search_budget: usize,
}

impl Default for FiniteViabilityModel {
    fn default() -> Self {
        Self {
            production_states: BTreeMap::new(),
            organization_states: BTreeMap::new(),
            boundary_states: BTreeMap::new(),
            realization: BTreeMap::new(),
            boundary_production: BTreeMap::new(),
            required_boundary_classes: BTreeSet::new(),
            admissible_boundaries: BTreeSet::new(),
            variables: BTreeMap::new(),
            constraints: Vec::new(),
            search_budget: 10_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WitnessStatus {
    Pass,
    Fail,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViabilityFindingCode {
    EmptyViableProductionStates,
    EmptyViableOrganizationStates,
    StateMapKeyMismatch,
    MissingRealization,
    UnknownRealizationSource,
    UnknownRealizationTarget,
    NonViableRealizationTarget,
    RealizationNotWellDefined,
    RealizationNotInjective,
    RealizationNotSurjective,
    EmptyRequiredBoundaryClasses,
    MissingBoundaryProduction,
    UnknownBoundarySource,
    UnknownBoundaryTarget,
    NonViableBoundaryTarget,
    BoundaryClassNotWellDefined,
    RequiredBoundaryClassMissing,
    EmptyAdmissibilitySet,
    UnknownAdmissibleBoundary,
    InadmissibleProducedBoundary,
    EmptyVariableSet,
    EmptyVariableDomain,
    EmptyConstraintSet,
    DuplicateConstraintId,
    EmptyConstraintScope,
    DuplicateVariableInScope,
    UnknownConstraintVariable,
    EmptyLocalRelation,
    TupleArityMismatch,
    UnknownValueInRelation,
    SearchBudgetExhausted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViabilityFinding {
    pub code: ViabilityFindingCode,
    pub subject: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteViabilityReport {
    pub ap1_realization: Option<WitnessStatus>,
    pub ap2_boundary_production: Option<WitnessStatus>,
    pub ap3_admissibility: Option<WitnessStatus>,
    pub ap4_global_viability: Option<WitnessStatus>,
    pub locally_satisfiable_patch_count: usize,
    pub local_patch_count: usize,
    /// Some(true) only when every local relation is non-empty and the bounded
    /// global search exhaustively proves that no gluing exists.
    pub local_to_global_obstruction: Option<bool>,
    pub search_nodes_examined: usize,
    pub findings: Vec<ViabilityFinding>,
}

impl FiniteViabilityReport {
    /// Qualification is relative to this finite encoding only. It is not a
    /// claim of biological autopoiesis or a proof in arbitrary topos semantics.
    pub fn qualifies_finite_surrogate(&self) -> bool {
        self.ap1_realization == Some(WitnessStatus::Pass)
            && self.ap2_boundary_production == Some(WitnessStatus::Pass)
            && self.ap3_admissibility == Some(WitnessStatus::Pass)
            && self.ap4_global_viability == Some(WitnessStatus::Pass)
            && self.findings.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchResult {
    Found,
    Exhausted,
    NoSolution,
}

fn finding(
    report: &mut FiniteViabilityReport,
    code: ViabilityFindingCode,
    subject: Option<String>,
    detail: impl Into<String>,
) {
    report.findings.push(ViabilityFinding {
        code,
        subject,
        detail: detail.into(),
    });
}

fn class_map_is_bijective(source_to_target: &BTreeMap<ViabilityClassId, ViabilityClassId>) -> bool {
    let mut targets = BTreeSet::new();
    source_to_target
        .values()
        .all(|target| targets.insert(target.clone()))
}

fn partial_constraints_compatible(
    constraints: &[LocalConstraint],
    assignment: &BTreeMap<VariableId, ValueId>,
) -> bool {
    constraints.iter().all(|constraint| {
        constraint.allowed_assignments.iter().any(|tuple| {
            constraint
                .scope
                .iter()
                .zip(tuple.iter())
                .all(|(variable, value)| {
                    assignment
                        .get(variable)
                        .is_none_or(|assigned| assigned == value)
                })
        })
    })
}

fn search_global_regime(
    index: usize,
    variables: &[(VariableId, Vec<ValueId>)],
    constraints: &[LocalConstraint],
    assignment: &mut BTreeMap<VariableId, ValueId>,
    budget: usize,
    nodes_examined: &mut usize,
) -> SearchResult {
    if *nodes_examined >= budget {
        return SearchResult::Exhausted;
    }
    *nodes_examined += 1;

    if index == variables.len() {
        return if partial_constraints_compatible(constraints, assignment) {
            SearchResult::Found
        } else {
            SearchResult::NoSolution
        };
    }

    let (variable, domain) = &variables[index];
    for value in domain {
        assignment.insert(variable.clone(), value.clone());
        let result = if partial_constraints_compatible(constraints, assignment) {
            search_global_regime(
                index + 1,
                variables,
                constraints,
                assignment,
                budget,
                nodes_examined,
            )
        } else {
            SearchResult::NoSolution
        };
        assignment.remove(variable);

        match result {
            SearchResult::Found => return SearchResult::Found,
            SearchResult::Exhausted => return SearchResult::Exhausted,
            SearchResult::NoSolution => {}
        }
    }
    SearchResult::NoSolution
}

/// Evaluate four finite witnesses independently. This is an explicit finite
/// surrogate, not a categorical implementation of AP1–AP4.
pub fn evaluate_finite_viability_model(model: &FiniteViabilityModel) -> FiniteViabilityReport {
    let mut report = FiniteViabilityReport::default();

    // AP1-style witness: realization induces a bijection on viable classes.
    let mut ap1_ok = true;
    let viable_production: Vec<_> = model
        .production_states
        .iter()
        .filter(|(_, state)| state.viable)
        .collect();
    let viable_organization_classes: BTreeSet<_> = model
        .organization_states
        .values()
        .filter(|state| state.viable)
        .map(|state| state.viability_class.clone())
        .collect();

    if viable_production.is_empty() {
        ap1_ok = false;
        finding(
            &mut report,
            ViabilityFindingCode::EmptyViableProductionStates,
            None,
            "no viable production states were declared",
        );
    }
    if viable_organization_classes.is_empty() {
        ap1_ok = false;
        finding(
            &mut report,
            ViabilityFindingCode::EmptyViableOrganizationStates,
            None,
            "no viable organization equivalence classes were declared",
        );
    }

    let mut induced_realization = BTreeMap::new();
    for (key, state) in &model.production_states {
        if key.0 != state.id {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::StateMapKeyMismatch,
                Some(key.0.clone()),
                "production-state map key differs from the embedded state ID",
            );
        }
    }
    for (key, state) in &model.organization_states {
        if key.0 != state.id {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::StateMapKeyMismatch,
                Some(key.0.clone()),
                "organization-state map key differs from the embedded state ID",
            );
        }
    }

    for (source, target) in &model.realization {
        if !model.production_states.contains_key(source) {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::UnknownRealizationSource,
                Some(source.0.clone()),
                "realization map contains an unknown production-state source",
            );
        }
        if !model.organization_states.contains_key(target) {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::UnknownRealizationTarget,
                Some(source.0.clone()),
                "realization map contains an unknown organization-state target",
            );
        }
    }

    for (production_id, production_state) in &viable_production {
        let Some(organization_id) = model.realization.get(*production_id) else {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::MissingRealization,
                Some(production_id.0.clone()),
                "viable production state has no realization",
            );
            continue;
        };
        let Some(organization_state) = model.organization_states.get(organization_id) else {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::UnknownRealizationTarget,
                Some(production_id.0.clone()),
                "realization references an unknown organization state",
            );
            continue;
        };
        if !organization_state.viable {
            ap1_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::NonViableRealizationTarget,
                Some(production_id.0.clone()),
                "viable production state realizes to a non-viable organization state",
            );
            continue;
        }
        if let Some(previous) = induced_realization.insert(
            production_state.viability_class.clone(),
            organization_state.viability_class.clone(),
        ) {
            if previous != organization_state.viability_class {
                ap1_ok = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::RealizationNotWellDefined,
                    Some(production_id.0.clone()),
                    "one viable production class maps to multiple organization classes",
                );
            }
        }
    }
    if !class_map_is_bijective(&induced_realization) {
        ap1_ok = false;
        finding(
            &mut report,
            ViabilityFindingCode::RealizationNotInjective,
            None,
            "distinct viable production classes map to the same organization class",
        );
    }
    let mapped_targets: BTreeSet<_> = induced_realization.values().cloned().collect();
    if mapped_targets != viable_organization_classes {
        ap1_ok = false;
        finding(
            &mut report,
            ViabilityFindingCode::RealizationNotSurjective,
            None,
            "realization does not cover exactly the viable organization classes",
        );
    }
    report.ap1_realization = Some(if ap1_ok {
        WitnessStatus::Pass
    } else {
        WitnessStatus::Fail
    });

    // AP2-style witness: viable production classes produce viable boundary classes.
    let mut ap2_ok = true;
    if model.required_boundary_classes.is_empty() {
        ap2_ok = false;
        finding(
            &mut report,
            ViabilityFindingCode::EmptyRequiredBoundaryClasses,
            None,
            "no required viable boundary class was declared",
        );
    }
    for (key, state) in &model.boundary_states {
        if key.0 != state.id {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::StateMapKeyMismatch,
                Some(key.0.clone()),
                "boundary-state map key differs from the embedded state ID",
            );
        }
    }
    for (source, target) in &model.boundary_production {
        if !model.production_states.contains_key(source) {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::UnknownBoundarySource,
                Some(source.0.clone()),
                "boundary-production map contains an unknown production-state source",
            );
        }
        if !model.boundary_states.contains_key(target) {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::UnknownBoundaryTarget,
                Some(source.0.clone()),
                "boundary-production map contains an unknown boundary-state target",
            );
        }
    }
    let mut induced_boundary = BTreeMap::new();
    let mut represented_boundary_classes = BTreeSet::new();
    for (production_id, production_state) in &viable_production {
        let Some(boundary_id) = model.boundary_production.get(*production_id) else {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::MissingBoundaryProduction,
                Some(production_id.0.clone()),
                "viable production state has no boundary-production witness",
            );
            continue;
        };
        let Some(boundary_state) = model.boundary_states.get(boundary_id) else {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::UnknownBoundaryTarget,
                Some(production_id.0.clone()),
                "boundary-production map references an unknown boundary state",
            );
            continue;
        };
        if !boundary_state.viable {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::NonViableBoundaryTarget,
                Some(production_id.0.clone()),
                "viable production state produces a non-viable boundary state",
            );
            continue;
        }
        represented_boundary_classes.insert(boundary_state.viability_class.clone());
        if let Some(previous) = induced_boundary.insert(
            production_state.viability_class.clone(),
            boundary_state.viability_class.clone(),
        ) {
            if previous != boundary_state.viability_class {
                ap2_ok = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::BoundaryClassNotWellDefined,
                    Some(production_id.0.clone()),
                    "one viable production class produces multiple boundary classes",
                );
            }
        }
    }
    for required in &model.required_boundary_classes {
        if !represented_boundary_classes.contains(required) {
            ap2_ok = false;
            finding(
                &mut report,
                ViabilityFindingCode::RequiredBoundaryClassMissing,
                Some(required.0.clone()),
                "a required viable boundary class is not produced",
            );
        }
    }
    report.ap2_boundary_production = Some(if ap2_ok {
        WitnessStatus::Pass
    } else {
        WitnessStatus::Fail
    });

    // AP3-style witness: every boundary produced by a viable process is admissible.
    let mut ap3_ok = true;
    if model.admissible_boundaries.is_empty() {
        ap3_ok = false;
        finding(
            &mut report,
            ViabilityFindingCode::EmptyAdmissibilitySet,
            None,
            "the admissibility predicate contains no boundary states",
        );
    }
    for boundary_id in &model.admissible_boundaries {
        match model.boundary_states.get(boundary_id) {
            Some(state) if state.viable => {}
            _ => {
                ap3_ok = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::UnknownAdmissibleBoundary,
                    Some(boundary_id.0.clone()),
                    "admissibility references an unknown or non-viable boundary state",
                );
            }
        }
    }
    for production_id in viable_production.iter().map(|(id, _)| *id) {
        match model.boundary_production.get(production_id) {
            None => {
                ap3_ok = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::MissingBoundaryProduction,
                    Some(production_id.0.clone()),
                    "admissibility cannot be established without a boundary-production witness",
                );
            }
            Some(boundary_id) if !model.boundary_states.contains_key(boundary_id) => {
                ap3_ok = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::UnknownBoundaryTarget,
                    Some(production_id.0.clone()),
                    "produced boundary target is not declared in the finite model",
                );
            }
            Some(boundary_id) if !model.admissible_boundaries.contains(boundary_id) => {
                ap3_ok = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::InadmissibleProducedBoundary,
                    Some(production_id.0.clone()),
                    "viable production produces a boundary outside the frozen admissibility predicate",
                );
            }
            Some(_) => {}
        }
    }
    report.ap3_admissibility = Some(if ap3_ok {
        WitnessStatus::Pass
    } else {
        WitnessStatus::Fail
    });

    // AP4-style witness: a non-empty globally consistent assignment exists.
    let mut ap4_valid = true;
    if model.variables.is_empty() {
        ap4_valid = false;
        finding(
            &mut report,
            ViabilityFindingCode::EmptyVariableSet,
            None,
            "global viability requires a non-empty finite variable set",
        );
    }
    if model.constraints.is_empty() {
        ap4_valid = false;
        finding(
            &mut report,
            ViabilityFindingCode::EmptyConstraintSet,
            None,
            "global viability requires at least one registered local constraint",
        );
    }
    let mut constraint_ids = BTreeSet::new();
    let mut local_ok_count = 0;
    for constraint in &model.constraints {
        if !constraint_ids.insert(constraint.id.clone()) {
            ap4_valid = false;
            finding(
                &mut report,
                ViabilityFindingCode::DuplicateConstraintId,
                Some(constraint.id.clone()),
                "local constraint ID is duplicated",
            );
        }
        let mut relation_valid = true;
        if constraint.scope.is_empty() {
            relation_valid = false;
            ap4_valid = false;
            finding(
                &mut report,
                ViabilityFindingCode::EmptyConstraintScope,
                Some(constraint.id.clone()),
                "local constraint scope is empty",
            );
        }
        let mut seen_variables = BTreeSet::new();
        for variable in &constraint.scope {
            if !seen_variables.insert(variable.clone()) {
                relation_valid = false;
                ap4_valid = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::DuplicateVariableInScope,
                    Some(constraint.id.clone()),
                    "local constraint repeats a variable in its scope",
                );
            }
            if !model.variables.contains_key(variable) {
                relation_valid = false;
                ap4_valid = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::UnknownConstraintVariable,
                    Some(constraint.id.clone()),
                    "local constraint references an unknown variable",
                );
            }
        }
        if constraint.allowed_assignments.is_empty() {
            ap4_valid = false;
            finding(
                &mut report,
                ViabilityFindingCode::EmptyLocalRelation,
                Some(constraint.id.clone()),
                "local patch has no allowed assignment",
            );
            continue;
        }

        for variable in &constraint.scope {
            if model
                .variables
                .get(variable)
                .is_some_and(BTreeSet::is_empty)
            {
                relation_valid = false;
            }
        }
        for tuple in &constraint.allowed_assignments {
            if tuple.len() != constraint.scope.len() {
                relation_valid = false;
                ap4_valid = false;
                finding(
                    &mut report,
                    ViabilityFindingCode::TupleArityMismatch,
                    Some(constraint.id.clone()),
                    "allowed tuple arity differs from the local constraint scope",
                );
                continue;
            }
            for (variable, value) in constraint.scope.iter().zip(tuple.iter()) {
                if model
                    .variables
                    .get(variable)
                    .is_some_and(|domain| !domain.contains(value))
                {
                    relation_valid = false;
                    ap4_valid = false;
                    finding(
                        &mut report,
                        ViabilityFindingCode::UnknownValueInRelation,
                        Some(constraint.id.clone()),
                        "allowed tuple contains a value outside the variable domain",
                    );
                }
            }
        }
        if relation_valid {
            local_ok_count += 1;
        }
    }
    for (variable, domain) in &model.variables {
        if domain.is_empty() {
            ap4_valid = false;
            finding(
                &mut report,
                ViabilityFindingCode::EmptyVariableDomain,
                Some(variable.0.clone()),
                "finite variable has an empty domain",
            );
        }
    }
    report.local_patch_count = model.constraints.len();
    report.locally_satisfiable_patch_count = local_ok_count;

    if ap4_valid && local_ok_count == model.constraints.len() {
        let variables: Vec<_> = model
            .variables
            .iter()
            .map(|(id, domain)| (id.clone(), domain.iter().cloned().collect()))
            .collect();
        let mut assignment = BTreeMap::new();
        let mut nodes = 0;
        let result = if model.search_budget == 0 {
            SearchResult::Exhausted
        } else {
            search_global_regime(
                0,
                &variables,
                &model.constraints,
                &mut assignment,
                model.search_budget,
                &mut nodes,
            )
        };
        report.search_nodes_examined = nodes;
        report.ap4_global_viability = Some(match result {
            SearchResult::Found => WitnessStatus::Pass,
            SearchResult::NoSolution => WitnessStatus::Fail,
            SearchResult::Exhausted => {
                finding(
                    &mut report,
                    ViabilityFindingCode::SearchBudgetExhausted,
                    None,
                    "global search reached its declared node budget; non-existence is not established",
                );
                WitnessStatus::Inconclusive
            }
        });
        report.local_to_global_obstruction = match result {
            SearchResult::Found => Some(false),
            SearchResult::NoSolution => Some(true),
            SearchResult::Exhausted => None,
        };
    } else {
        report.ap4_global_viability = Some(WitnessStatus::Fail);
        report.local_to_global_obstruction = None;
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id<T: From<&'static str>>(value: &'static str) -> T {
        value.into()
    }

    fn healthy_model() -> FiniteViabilityModel {
        let mut model = FiniteViabilityModel::default();
        model
            .production_states
            .insert(id("p0"), FiniteState::new("p0", "prod-class", true));
        model
            .production_states
            .insert(id("p1"), FiniteState::new("p1", "prod-class", true));
        model
            .organization_states
            .insert(id("o0"), FiniteState::new("o0", "org-class", true));
        model
            .organization_states
            .insert(id("o1"), FiniteState::new("o1", "org-class", true));
        model
            .boundary_states
            .insert(id("b0"), FiniteState::new("b0", "boundary-class", true));
        model.realization.insert(id("p0"), id("o0"));
        model.realization.insert(id("p1"), id("o1"));
        model.boundary_production.insert(id("p0"), id("b0"));
        model.boundary_production.insert(id("p1"), id("b0"));
        model.required_boundary_classes.insert(id("boundary-class"));
        model.admissible_boundaries.insert(id("b0"));
        model
            .variables
            .insert(id("x"), [id("zero"), id("one")].into_iter().collect());
        model.constraints.push(LocalConstraint::new(
            "x-is-one",
            vec![id("x")],
            vec![vec![id("one")]],
        ));
        model
    }

    #[test]
    fn healthy_finite_model_passes_all_four_witnesses() {
        let report = evaluate_finite_viability_model(&healthy_model());
        assert!(report.qualifies_finite_surrogate(), "{:?}", report.findings);
        assert_eq!(report.ap1_realization, Some(WitnessStatus::Pass));
        assert_eq!(report.ap2_boundary_production, Some(WitnessStatus::Pass));
        assert_eq!(report.ap3_admissibility, Some(WitnessStatus::Pass));
        assert_eq!(report.ap4_global_viability, Some(WitnessStatus::Pass));
        assert_eq!(report.local_to_global_obstruction, Some(false));
    }

    #[test]
    fn non_well_defined_realization_fails_ap1() {
        let mut model = healthy_model();
        model
            .organization_states
            .insert(id("o2"), FiniteState::new("o2", "other-org-class", true));
        model.realization.insert(id("p1"), id("o2"));
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(report.ap1_realization, Some(WitnessStatus::Fail));
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ViabilityFindingCode::RealizationNotWellDefined)
        );
    }

    #[test]
    fn missing_boundary_production_fails_ap2() {
        let mut model = healthy_model();
        model.boundary_production.remove(&id("p1"));
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(report.ap2_boundary_production, Some(WitnessStatus::Fail));
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ViabilityFindingCode::MissingBoundaryProduction)
        );
    }

    #[test]
    fn inadmissible_boundary_fails_ap3() {
        let mut model = healthy_model();
        model.admissible_boundaries.clear();
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(report.ap3_admissibility, Some(WitnessStatus::Fail));
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ViabilityFindingCode::InadmissibleProducedBoundary)
        );
    }

    #[test]
    fn local_patches_can_be_satisfiable_without_a_global_regime() {
        let mut model = healthy_model();
        model.variables.clear();
        model.constraints.clear();
        for name in ["x", "y", "z"] {
            model
                .variables
                .insert(id(name), [id("0"), id("1")].into_iter().collect());
        }
        model.constraints = vec![
            LocalConstraint::new(
                "xy-equal",
                vec![id("x"), id("y")],
                vec![vec![id("0"), id("0")], vec![id("1"), id("1")]],
            ),
            LocalConstraint::new(
                "yz-equal",
                vec![id("y"), id("z")],
                vec![vec![id("0"), id("0")], vec![id("1"), id("1")]],
            ),
            LocalConstraint::new(
                "xz-different",
                vec![id("x"), id("z")],
                vec![vec![id("0"), id("1")], vec![id("1"), id("0")]],
            ),
        ];
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(report.locally_satisfiable_patch_count, 3);
        assert_eq!(report.ap4_global_viability, Some(WitnessStatus::Fail));
        assert_eq!(report.local_to_global_obstruction, Some(true));
    }

    #[test]
    fn search_budget_exhaustion_is_inconclusive_not_failure() {
        let mut model = healthy_model();
        model.variables.clear();
        model.constraints.clear();
        model
            .variables
            .insert(id("x"), [id("0"), id("1")].into_iter().collect());
        model
            .variables
            .insert(id("y"), [id("0"), id("1")].into_iter().collect());
        model.constraints = vec![
            LocalConstraint::new(
                "requires-equal",
                vec![id("x"), id("y")],
                vec![vec![id("0"), id("0")], vec![id("1"), id("1")]],
            ),
            LocalConstraint::new(
                "requires-different",
                vec![id("x"), id("y")],
                vec![vec![id("0"), id("1")], vec![id("1"), id("0")]],
            ),
        ];
        model.search_budget = 1;
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(
            report.ap4_global_viability,
            Some(WitnessStatus::Inconclusive)
        );
        assert_eq!(report.local_to_global_obstruction, None);
    }

    #[test]
    fn empty_state_universe_cannot_vacuously_pass() {
        let model = FiniteViabilityModel::default();
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(report.ap1_realization, Some(WitnessStatus::Fail));
        assert_eq!(report.ap2_boundary_production, Some(WitnessStatus::Fail));
        assert_eq!(report.ap3_admissibility, Some(WitnessStatus::Fail));
        assert_eq!(report.ap4_global_viability, Some(WitnessStatus::Fail));
        assert!(!report.qualifies_finite_surrogate());
    }

    #[test]
    fn malformed_local_relation_fails_closed() {
        let mut model = healthy_model();
        model.constraints[0] = LocalConstraint::new(
            "wrong-arity",
            vec![id("x")],
            vec![vec![id("zero"), id("one")]],
        );
        let report = evaluate_finite_viability_model(&model);
        assert_eq!(report.ap4_global_viability, Some(WitnessStatus::Fail));
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ViabilityFindingCode::TupleArityMismatch)
        );
    }
}
