// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Economic Theory Intermediate Representation (ETIR) v1.
//!
//! ETIR describes theory-neutral variables, mechanisms, and decomposed claims.
//! Model adapters declare which parts they implement. A model family therefore
//! does not own the scientific claim it computes.

use std::collections::BTreeSet;

use crate::error::{EconomicsError, Result};
use crate::ontology::{ClaimId, EconomicVariable, MechanismId, ModelId, TheoryId, VariableId};
use crate::theory::EconomicClaim;

pub const ETIR_SCHEMA_VERSION: u16 = 1;

fn ensure_unique_refs<T: Ord>(
    values: impl IntoIterator<Item = T>,
    context: &'static str,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(EconomicsError::InvalidParameter { context });
        }
    }
    Ok(())
}

/// A mechanism connects declared state variables without prescribing how a
/// particular simulator or estimator must implement the transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MechanismSpec {
    id: MechanismId,
    description: String,
    inputs: Vec<VariableId>,
    outputs: Vec<VariableId>,
}

impl MechanismSpec {
    pub fn new(
        id: MechanismId,
        description: impl Into<String>,
        inputs: Vec<VariableId>,
        outputs: Vec<VariableId>,
    ) -> Result<Self> {
        let description = description.into();
        if description.trim().is_empty() {
            return Err(EconomicsError::InvalidParameter {
                context: "mechanism description",
            });
        }
        if outputs.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "mechanism outputs",
            });
        }
        ensure_unique_refs(inputs.iter(), "duplicate mechanism input")?;
        ensure_unique_refs(outputs.iter(), "duplicate mechanism output")?;
        Ok(Self {
            id,
            description,
            inputs,
            outputs,
        })
    }

    pub fn id(&self) -> &MechanismId {
        &self.id
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn inputs(&self) -> &[VariableId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[VariableId] {
        &self.outputs
    }
}

/// Computational paradigms are composable labels, not mutually exclusive
/// schools. An ABM may also be stock-flow-consistent and network-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelParadigm {
    AgentBased,
    StockFlowConsistent,
    StructuralCausal,
    Equilibrium,
    Econometric,
    Network,
    SystemDynamics,
    MachineLearning,
}

/// A model implementation declaration bound to an ETIR theory. This is
/// non-authorizing metadata: declaration does not prove runtime conformance,
/// causal validity, empirical support, or policy authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAdapterDeclaration {
    id: ModelId,
    theory: TheoryId,
    paradigms: Vec<ModelParadigm>,
    implemented_claims: Vec<ClaimId>,
    predicted_variables: Vec<VariableId>,
}

impl ModelAdapterDeclaration {
    pub fn new(
        id: ModelId,
        theory: TheoryId,
        paradigms: Vec<ModelParadigm>,
        implemented_claims: Vec<ClaimId>,
        predicted_variables: Vec<VariableId>,
    ) -> Result<Self> {
        if paradigms.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "model adapter paradigms",
            });
        }
        if implemented_claims.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "model adapter claims",
            });
        }
        if predicted_variables.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "model adapter predicted variables",
            });
        }
        ensure_unique_refs(paradigms.iter(), "duplicate model paradigm")?;
        ensure_unique_refs(implemented_claims.iter(), "duplicate adapter claim")?;
        ensure_unique_refs(
            predicted_variables.iter(),
            "duplicate adapter predicted variable",
        )?;
        Ok(Self {
            id,
            theory,
            paradigms,
            implemented_claims,
            predicted_variables,
        })
    }

    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn theory(&self) -> &TheoryId {
        &self.theory
    }

    pub fn paradigms(&self) -> &[ModelParadigm] {
        &self.paradigms
    }

    pub fn implemented_claims(&self) -> &[ClaimId] {
        &self.implemented_claims
    }

    pub fn predicted_variables(&self) -> &[VariableId] {
        &self.predicted_variables
    }
}

/// A validated theory-neutral economic theory graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoryIr {
    schema_version: u16,
    id: TheoryId,
    variables: Vec<EconomicVariable>,
    mechanisms: Vec<MechanismSpec>,
    claims: Vec<EconomicClaim>,
}

impl TheoryIr {
    pub fn new(
        id: TheoryId,
        variables: Vec<EconomicVariable>,
        mechanisms: Vec<MechanismSpec>,
        claims: Vec<EconomicClaim>,
    ) -> Result<Self> {
        if variables.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "ETIR variables",
            });
        }
        if claims.is_empty() {
            return Err(EconomicsError::EmptyInput {
                context: "ETIR claims",
            });
        }

        ensure_unique_refs(
            variables.iter().map(EconomicVariable::id),
            "duplicate ETIR variable",
        )?;
        ensure_unique_refs(
            mechanisms.iter().map(MechanismSpec::id),
            "duplicate ETIR mechanism",
        )?;
        ensure_unique_refs(claims.iter().map(EconomicClaim::id), "duplicate ETIR claim")?;

        let variable_ids: BTreeSet<&VariableId> = variables.iter().map(EconomicVariable::id).collect();
        let mechanism_ids: BTreeSet<&MechanismId> = mechanisms.iter().map(MechanismSpec::id).collect();

        for mechanism in &mechanisms {
            for variable in mechanism.inputs().iter().chain(mechanism.outputs()) {
                if !variable_ids.contains(variable) {
                    return Err(EconomicsError::InvalidParameter {
                        context: "mechanism references undeclared ETIR variable",
                    });
                }
            }
        }
        for claim in &claims {
            for variable in claim.referenced_variables() {
                if !variable_ids.contains(variable) {
                    return Err(EconomicsError::InvalidParameter {
                        context: "claim references undeclared ETIR variable",
                    });
                }
            }
            for mechanism in claim.referenced_mechanisms() {
                if !mechanism_ids.contains(mechanism) {
                    return Err(EconomicsError::InvalidParameter {
                        context: "claim references undeclared ETIR mechanism",
                    });
                }
            }

            if let EconomicClaim::Empirical(empirical) = claim {
                if empirical.mode() == crate::theory::EmpiricalClaimMode::Mechanistic {
                    let mut claim_mechanisms = Vec::with_capacity(empirical.mechanisms().len());
                    for mechanism_id in empirical.mechanisms() {
                        let Some(spec) = mechanisms.iter().find(|item| item.id() == mechanism_id)
                        else {
                            return Err(EconomicsError::InvalidParameter {
                                context: "claim references undeclared ETIR mechanism",
                            });
                        };
                        claim_mechanisms.push(spec);
                    }

                    for prediction in empirical.predictions() {
                        if !claim_mechanisms
                            .iter()
                            .any(|mechanism| mechanism.outputs().contains(prediction.outcome()))
                        {
                            return Err(EconomicsError::InvalidParameter {
                                context: "mechanistic claim outcome not produced by claimed mechanism",
                            });
                        }
                    }

                    let mut required_variables: BTreeSet<&VariableId> = empirical
                        .predictions()
                        .iter()
                        .map(crate::theory::Prediction::outcome)
                        .collect();
                    let mut connected_mechanisms: BTreeSet<&MechanismId> = BTreeSet::new();
                    loop {
                        let mut changed = false;
                        for mechanism in &claim_mechanisms {
                            if connected_mechanisms.contains(mechanism.id()) {
                                continue;
                            }
                            if mechanism
                                .outputs()
                                .iter()
                                .any(|output| required_variables.contains(output))
                            {
                                connected_mechanisms.insert(mechanism.id());
                                required_variables.extend(mechanism.inputs());
                                changed = true;
                            }
                        }
                        if !changed {
                            break;
                        }
                    }

                    if connected_mechanisms.len() != claim_mechanisms.len() {
                        return Err(EconomicsError::InvalidParameter {
                            context: "mechanistic claim contains disconnected mechanism",
                        });
                    }
                }
            }
        }

        Ok(Self {
            schema_version: ETIR_SCHEMA_VERSION,
            id,
            variables,
            mechanisms,
            claims,
        })
    }

    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn id(&self) -> &TheoryId {
        &self.id
    }

    pub fn variables(&self) -> &[EconomicVariable] {
        &self.variables
    }

    pub fn mechanisms(&self) -> &[MechanismSpec] {
        &self.mechanisms
    }

    pub fn claims(&self) -> &[EconomicClaim] {
        &self.claims
    }

    /// Validate that a concrete model adapter is a bounded declaration against
    /// this theory. It remains a declaration, not evidence that implementation
    /// behavior actually conforms to the declared claim or mechanism.
    pub fn validate_adapter(&self, adapter: &ModelAdapterDeclaration) -> Result<()> {
        if adapter.theory != self.id {
            return Err(EconomicsError::InvalidParameter {
                context: "model adapter references different theory",
            });
        }
        for claim_id in &adapter.implemented_claims {
            let Some(claim) = self.claims.iter().find(|claim| claim.id() == claim_id) else {
                return Err(EconomicsError::InvalidParameter {
                    context: "model adapter references unknown claim",
                });
            };
            if matches!(claim, EconomicClaim::Normative(_)) {
                return Err(EconomicsError::InvalidParameter {
                    context: "model adapter cannot implement normative proposition",
                });
            }
            if let EconomicClaim::Empirical(empirical) = claim {
                for prediction in empirical.predictions() {
                    if !adapter.predicted_variables.contains(prediction.outcome()) {
                        return Err(EconomicsError::InvalidParameter {
                            context: "model adapter omits empirical claim outcome",
                        });
                    }
                }
            }
        }
        let variables: BTreeSet<&VariableId> = self.variables.iter().map(EconomicVariable::id).collect();
        for variable in &adapter.predicted_variables {
            if !variables.contains(variable) {
                return Err(EconomicsError::InvalidParameter {
                    context: "model adapter references unknown predicted variable",
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::{PredictionId, StateDomain, UnitId};
    use crate::theory::{
        EmpiricalClaim, EmpiricalClaimMode, FalsificationCriterion, NormativeProposition,
        Prediction, ResponseDirection,
    };

    fn employment() -> EconomicVariable {
        EconomicVariable::new(
            VariableId::new("labor:employment").unwrap(),
            StateDomain::Institutional,
            UnitId::new("persons").unwrap(),
            "Employment",
        )
        .unwrap()
    }

    fn demand() -> EconomicVariable {
        EconomicVariable::new(
            VariableId::new("macro:nominal_demand").unwrap(),
            StateDomain::Financial,
            UnitId::new("currency_atoms_per_period").unwrap(),
            "Nominal aggregate demand",
        )
        .unwrap()
    }

    fn prices() -> EconomicVariable {
        EconomicVariable::new(
            VariableId::new("prices:index").unwrap(),
            StateDomain::Financial,
            UnitId::new("index").unwrap(),
            "Price index",
        )
        .unwrap()
    }

    fn theory() -> TheoryIr {
        let mechanism_id = MechanismId::new("nominal_rigidity").unwrap();
        let claim = EconomicClaim::Empirical(
            EmpiricalClaim::new(
                ClaimId::new("demand_employment").unwrap(),
                "Demand contraction lowers employment under nominal rigidity.",
                EmpiricalClaimMode::Mechanistic,
                "unused_capacity_and_slow_adjustment",
                vec![mechanism_id.clone()],
                vec![Prediction::new(
                    PredictionId::new("employment_response").unwrap(),
                    VariableId::new("labor:employment").unwrap(),
                    ResponseDirection::Decrease,
                    "4_quarters",
                    None,
                )
                .unwrap()],
                vec![FalsificationCriterion::new(
                    PredictionId::new("employment_response").unwrap(),
                    "Comparable contractions do not reduce employment.",
                )
                .unwrap()],
            )
            .unwrap(),
        );
        TheoryIr::new(
            TheoryId::new("macro:rigidity_v1").unwrap(),
            vec![demand(), employment()],
            vec![MechanismSpec::new(
                mechanism_id,
                "Nominal adjustment frictions transmit demand into real activity.",
                vec![VariableId::new("macro:nominal_demand").unwrap()],
                vec![VariableId::new("labor:employment").unwrap()],
            )
            .unwrap()],
            vec![claim],
        )
        .unwrap()
    }

    #[test]
    fn etir_rejects_unbound_mechanism_variables() {
        let result = TheoryIr::new(
            TheoryId::new("broken").unwrap(),
            vec![employment()],
            vec![MechanismSpec::new(
                MechanismId::new("m").unwrap(),
                "Broken mechanism",
                vec![VariableId::new("missing").unwrap()],
                vec![VariableId::new("labor:employment").unwrap()],
            )
            .unwrap()],
            vec![EconomicClaim::Normative(
                NormativeProposition::new(
                    ClaimId::new("n").unwrap(),
                    "Preserve agency.",
                    vec!["agency".into()],
                )
                .unwrap(),
            )],
        );
        assert!(result.is_err());
    }

    #[test]
    fn mechanistic_claim_rejects_decorative_disconnected_mechanisms() {
        let rigidity = MechanismId::new("nominal_rigidity").unwrap();
        let decorative = MechanismId::new("decorative_markup_story").unwrap();
        let claim = EconomicClaim::Empirical(
            EmpiricalClaim::new(
                ClaimId::new("mechanism_integrity").unwrap(),
                "Demand contraction lowers employment through a stated mechanism.",
                EmpiricalClaimMode::Mechanistic,
                "synthetic_scope",
                vec![rigidity.clone(), decorative.clone()],
                vec![Prediction::new(
                    PredictionId::new("employment_response").unwrap(),
                    VariableId::new("labor:employment").unwrap(),
                    ResponseDirection::Decrease,
                    "1_step",
                    None,
                )
                .unwrap()],
                vec![FalsificationCriterion::new(
                    PredictionId::new("employment_response").unwrap(),
                    "Employment does not decrease.",
                )
                .unwrap()],
            )
            .unwrap(),
        );

        let result = TheoryIr::new(
            TheoryId::new("decorative_mechanism").unwrap(),
            vec![demand(), employment(), prices()],
            vec![
                MechanismSpec::new(
                    rigidity,
                    "Demand transmits to employment.",
                    vec![VariableId::new("macro:nominal_demand").unwrap()],
                    vec![VariableId::new("labor:employment").unwrap()],
                )
                .unwrap(),
                MechanismSpec::new(
                    decorative,
                    "Unrelated price story.",
                    vec![],
                    vec![VariableId::new("prices:index").unwrap()],
                )
                .unwrap(),
            ],
            vec![claim],
        );
        assert!(result.is_err());
    }

    #[test]
    fn different_model_paradigms_can_bind_the_same_claim() {
        let theory = theory();
        let claim = ClaimId::new("demand_employment").unwrap();
        let output = VariableId::new("labor:employment").unwrap();
        let abm = ModelAdapterDeclaration::new(
            ModelId::new("abm:v1").unwrap(),
            theory.id().clone(),
            vec![ModelParadigm::AgentBased, ModelParadigm::StockFlowConsistent],
            vec![claim.clone()],
            vec![output.clone()],
        )
        .unwrap();
        let scm = ModelAdapterDeclaration::new(
            ModelId::new("scm:v1").unwrap(),
            theory.id().clone(),
            vec![ModelParadigm::StructuralCausal],
            vec![claim],
            vec![output],
        )
        .unwrap();

        theory.validate_adapter(&abm).unwrap();
        theory.validate_adapter(&scm).unwrap();
        assert_ne!(abm.paradigms(), scm.paradigms());
    }

    #[test]
    fn adapter_must_expose_every_empirical_claim_outcome() {
        let theory = theory();
        let adapter = ModelAdapterDeclaration::new(
            ModelId::new("incomplete:v1").unwrap(),
            theory.id().clone(),
            vec![ModelParadigm::AgentBased],
            vec![ClaimId::new("demand_employment").unwrap()],
            vec![VariableId::new("macro:nominal_demand").unwrap()],
        )
        .unwrap();

        assert!(theory.validate_adapter(&adapter).is_err());
    }

    #[test]
    fn adapter_cannot_implement_a_normative_proposition() {
        let normative_id = ClaimId::new("preserve_agency").unwrap();
        let theory = TheoryIr::new(
            TheoryId::new("normative_boundary").unwrap(),
            vec![employment()],
            vec![],
            vec![EconomicClaim::Normative(
                NormativeProposition::new(
                    normative_id.clone(),
                    "Institutions should preserve agency.",
                    vec!["agency".into()],
                )
                .unwrap(),
            )],
        )
        .unwrap();
        let adapter = ModelAdapterDeclaration::new(
            ModelId::new("optimizer:v1").unwrap(),
            theory.id().clone(),
            vec![ModelParadigm::MachineLearning],
            vec![normative_id],
            vec![VariableId::new("labor:employment").unwrap()],
        )
        .unwrap();

        assert!(theory.validate_adapter(&adapter).is_err());
    }

    #[test]
    fn adapter_cannot_claim_unknown_theory_content() {
        let theory = theory();
        let adapter = ModelAdapterDeclaration::new(
            ModelId::new("bad_adapter").unwrap(),
            theory.id().clone(),
            vec![ModelParadigm::Econometric],
            vec![ClaimId::new("not_in_theory").unwrap()],
            vec![VariableId::new("labor:employment").unwrap()],
        )
        .unwrap();
        assert!(theory.validate_adapter(&adapter).is_err());
    }
}
