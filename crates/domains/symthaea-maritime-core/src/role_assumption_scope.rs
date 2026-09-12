// SPDX-License-Identifier: AGPL-3.0-or-later
//! Derive which modeled assumptions each regenerative-lineage role actually depends on.
//!
//! Calibration should not rely on a caller manually asserting whether a dynamic
//! intervention changed the assumptions behind a static role theorem. This module
//! derives a role's transitive dependency/support scope from the exact viability
//! profile, Genome, closure model, and flow-support graph, then intersects that
//! scope with an evidence-bound change set.

use crate::{
    RegenerativeClosureModel, RegenerativeFlowSupportV1, RegenerativeGenomeV1,
    RegenerativeLineageRoleV1, RegenerativeLineageViabilityError,
    RegenerativeLineageViabilityProfileV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const MAX_CHANGES: usize = 4096;
const MAX_BINDING_LEN: usize = 1024;
const MAX_ID_LEN: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeRoleAssumptionScopeV1 {
    pub role: RegenerativeLineageRoleV1,
    pub capability_ids: Vec<String>,
    pub dependency_ids: Vec<String>,
    pub external_input_bindings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeAssumptionChangeSetV1 {
    pub change_set_id: String,
    pub changed_dependency_ids: Vec<String>,
    pub changed_external_input_bindings: Vec<String>,
    pub evidence_binding: String,
}

impl RegenerativeAssumptionChangeSetV1 {
    pub fn validate(&self) -> Result<(), RegenerativeRoleAssumptionScopeError> {
        validate_id(&self.change_set_id)?;
        validate_binding(&self.evidence_binding)?;
        if self.changed_dependency_ids.len() > MAX_CHANGES
            || self.changed_external_input_bindings.len() > MAX_CHANGES
        {
            return Err(RegenerativeRoleAssumptionScopeError::TooManyChanges);
        }
        validate_ids(&self.changed_dependency_ids)?;
        validate_bindings(&self.changed_external_input_bindings)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeRoleAssumptionAssessmentV1 {
    pub role: RegenerativeLineageRoleV1,
    pub static_assumptions_held: bool,
    pub intersecting_changed_dependency_ids: Vec<String>,
    pub intersecting_changed_external_input_bindings: Vec<String>,
    pub change_set_id: String,
    pub change_evidence_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeRoleAssumptionScopeError {
    LineageProfileInvalid(RegenerativeLineageViabilityError),
    GenomeInvalid(crate::RegenerativeGenomeError),
    FlowSupportInvalid(crate::RegenerativeFlowSupportError),
    InvalidIdentifier,
    InvalidBinding,
    TooManyChanges,
    NonCanonicalIdentifiers,
    NonCanonicalBindings,
}

pub fn derive_regenerative_role_assumption_scopes(
    profile: &RegenerativeLineageViabilityProfileV1,
    genome: &RegenerativeGenomeV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<Vec<RegenerativeRoleAssumptionScopeV1>, RegenerativeRoleAssumptionScopeError> {
    profile
        .validate_against_model(model)
        .map_err(RegenerativeRoleAssumptionScopeError::LineageProfileInvalid)?;
    genome
        .validate_against_model(model)
        .map_err(RegenerativeRoleAssumptionScopeError::GenomeInvalid)?;
    support
        .validate_against_model(model)
        .map_err(RegenerativeRoleAssumptionScopeError::FlowSupportInvalid)?;

    let claims_by_dependency = support.claims.iter().fold(
        BTreeMap::<&str, Vec<_>>::new(),
        |mut claims, claim| {
            claims
                .entry(claim.dependency_id.as_str())
                .or_default()
                .push(claim);
            claims
        },
    );

    let mut scopes = Vec::new();
    for role in [
        RegenerativeLineageRoleV1::Operation,
        RegenerativeLineageRoleV1::SuccessorConstruction,
        RegenerativeLineageRoleV1::SuccessorQualification,
    ] {
        let capability_ids = role_capability_ids(profile, role).to_vec();
        let wanted_capabilities: BTreeSet<&str> =
            capability_ids.iter().map(String::as_str).collect();

        let mut dependency_ids = BTreeSet::new();
        let mut queue = VecDeque::new();
        for requirement in &genome.requirements {
            if wanted_capabilities.contains(requirement.capability_id.as_str())
                && dependency_ids.insert(requirement.baseline_dependency_id.clone())
            {
                queue.push_back(requirement.baseline_dependency_id.clone());
            }
        }

        let mut external_input_bindings = BTreeSet::new();
        while let Some(dependency_id) = queue.pop_front() {
            if let Some(claims) = claims_by_dependency.get(dependency_id.as_str()) {
                for claim in claims {
                    if let Some(binding) = &claim.external_input_binding {
                        external_input_bindings.insert(binding.clone());
                    }
                    for prerequisite_id in &claim.prerequisite_dependency_ids {
                        if dependency_ids.insert(prerequisite_id.clone()) {
                            queue.push_back(prerequisite_id.clone());
                        }
                    }
                }
            }
        }

        scopes.push(RegenerativeRoleAssumptionScopeV1 {
            role,
            capability_ids,
            dependency_ids: dependency_ids.into_iter().collect(),
            external_input_bindings: external_input_bindings.into_iter().collect(),
        });
    }
    Ok(scopes)
}

pub fn assess_regenerative_role_assumption_changes(
    scopes: &[RegenerativeRoleAssumptionScopeV1],
    changes: &RegenerativeAssumptionChangeSetV1,
) -> Result<Vec<RegenerativeRoleAssumptionAssessmentV1>, RegenerativeRoleAssumptionScopeError> {
    changes.validate()?;
    let changed_dependencies: BTreeSet<&str> =
        changes.changed_dependency_ids.iter().map(String::as_str).collect();
    let changed_external_inputs: BTreeSet<&str> = changes
        .changed_external_input_bindings
        .iter()
        .map(String::as_str)
        .collect();

    Ok(scopes
        .iter()
        .map(|scope| {
            let intersecting_changed_dependency_ids = scope
                .dependency_ids
                .iter()
                .filter(|id| changed_dependencies.contains(id.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            let intersecting_changed_external_input_bindings = scope
                .external_input_bindings
                .iter()
                .filter(|binding| changed_external_inputs.contains(binding.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            RegenerativeRoleAssumptionAssessmentV1 {
                role: scope.role,
                static_assumptions_held: intersecting_changed_dependency_ids.is_empty()
                    && intersecting_changed_external_input_bindings.is_empty(),
                intersecting_changed_dependency_ids,
                intersecting_changed_external_input_bindings,
                change_set_id: changes.change_set_id.clone(),
                change_evidence_binding: changes.evidence_binding.clone(),
            }
        })
        .collect())
}

fn role_capability_ids(
    profile: &RegenerativeLineageViabilityProfileV1,
    role: RegenerativeLineageRoleV1,
) -> &[String] {
    match role {
        RegenerativeLineageRoleV1::Operation => &profile.operational_capability_ids,
        RegenerativeLineageRoleV1::SuccessorConstruction => {
            &profile.successor_construction_capability_ids
        }
        RegenerativeLineageRoleV1::SuccessorQualification => {
            &profile.successor_qualification_capability_ids
        }
    }
}

fn validate_id(value: &str) -> Result<(), RegenerativeRoleAssumptionScopeError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeRoleAssumptionScopeError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeRoleAssumptionScopeError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeRoleAssumptionScopeError::InvalidBinding)
    } else {
        Ok(())
    }
}

fn validate_ids(values: &[String]) -> Result<(), RegenerativeRoleAssumptionScopeError> {
    for value in values {
        validate_id(value)?;
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(RegenerativeRoleAssumptionScopeError::NonCanonicalIdentifiers);
    }
    Ok(())
}

fn validate_bindings(values: &[String]) -> Result<(), RegenerativeRoleAssumptionScopeError> {
    for value in values {
        validate_binding(value)?;
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(RegenerativeRoleAssumptionScopeError::NonCanonicalBindings);
    }
    Ok(())
}
