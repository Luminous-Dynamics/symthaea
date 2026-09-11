// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit, evidence-bound dependency substitution for regenerative models.
//!
//! Substitution is diagnostic scenario construction only. It grants no physical
//! manufacturing, maintenance, procurement, or operating authority and is never
//! applied automatically by the closure model.

use crate::{DependencyGovernance, RegenerativeClosureModel};
use serde::{Deserialize, Serialize};

/// Current substitution-evidence schema version.
pub const REGENERATIVE_SUBSTITUTION_SCHEMA_V1: u8 = 1;

const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

/// Evidence describing one explicitly qualified dependency substitution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeSubstitutionEvidenceV1 {
    /// Exact schema version.
    pub schema_version: u8,
    /// Source closure-model identifier.
    pub source_model_id: String,
    /// Evidence/version binding of the source model.
    pub source_model_evidence_binding: String,
    /// Identifier assigned to the derived what-if model.
    pub derived_model_id: String,
    /// Evidence/version binding assigned to the derived model.
    pub derived_model_evidence_binding: String,
    /// Capability whose requirement is being substituted.
    pub capability_id: String,
    /// Exact evidence binding of that capability definition.
    pub capability_evidence_binding: String,
    /// Dependency currently required by the capability.
    pub baseline_dependency_id: String,
    /// Qualified alternative dependency already present in the source model.
    pub substitute_dependency_id: String,
    /// Opaque evidence binding for the engineering/qualification decision.
    pub qualification_binding: String,
}

impl RegenerativeSubstitutionEvidenceV1 {
    /// Validate standalone canonical shape.
    pub fn validate_shape(&self) -> Result<(), RegenerativeSubstitutionError> {
        if self.schema_version != REGENERATIVE_SUBSTITUTION_SCHEMA_V1 {
            return Err(RegenerativeSubstitutionError::UnsupportedSchema {
                schema_version: self.schema_version,
            });
        }
        for value in [
            &self.source_model_id,
            &self.derived_model_id,
            &self.capability_id,
            &self.baseline_dependency_id,
            &self.substitute_dependency_id,
        ] {
            validate_id(value)?;
        }
        for value in [
            &self.source_model_evidence_binding,
            &self.derived_model_evidence_binding,
            &self.capability_evidence_binding,
            &self.qualification_binding,
        ] {
            validate_binding(value)?;
        }
        if self.baseline_dependency_id == self.substitute_dependency_id {
            return Err(RegenerativeSubstitutionError::IdentitySubstitution);
        }
        if self.source_model_id == self.derived_model_id
            || self.source_model_evidence_binding == self.derived_model_evidence_binding
        {
            return Err(RegenerativeSubstitutionError::DerivedModelNotDistinct);
        }
        Ok(())
    }

    /// Build an explicit derived closure model with this one substitution applied.
    ///
    /// V1 permits only ordinary-to-ordinary substitutions of the same dependency
    /// kind. Dependencies marked `SafeguardedExternal` are deliberately excluded
    /// from this generic path so a substitution claim cannot weaken that boundary.
    pub fn apply_to_model(
        &self,
        source: &RegenerativeClosureModel,
    ) -> Result<RegenerativeClosureModel, RegenerativeSubstitutionError> {
        self.validate_shape()?;
        source
            .validate()
            .map_err(|_| RegenerativeSubstitutionError::InvalidSourceModel)?;

        if source.model_id != self.source_model_id
            || source.evidence_binding != self.source_model_evidence_binding
        {
            return Err(RegenerativeSubstitutionError::SourceModelMismatch);
        }

        let capability = source
            .capabilities
            .iter()
            .find(|capability| capability.capability_id == self.capability_id)
            .ok_or(RegenerativeSubstitutionError::CapabilityNotFound)?;
        if capability.evidence_binding != self.capability_evidence_binding {
            return Err(RegenerativeSubstitutionError::CapabilityBindingMismatch);
        }
        if !capability
            .dependency_ids
            .contains(&self.baseline_dependency_id)
        {
            return Err(RegenerativeSubstitutionError::BaselineNotRequired);
        }
        if capability
            .dependency_ids
            .contains(&self.substitute_dependency_id)
        {
            return Err(RegenerativeSubstitutionError::SubstituteAlreadyRequired);
        }

        let baseline = source
            .dependencies
            .iter()
            .find(|dependency| dependency.dependency_id == self.baseline_dependency_id)
            .ok_or(RegenerativeSubstitutionError::BaselineDependencyNotFound)?;
        let substitute = source
            .dependencies
            .iter()
            .find(|dependency| dependency.dependency_id == self.substitute_dependency_id)
            .ok_or(RegenerativeSubstitutionError::SubstituteDependencyNotFound)?;

        if baseline.governance != DependencyGovernance::Ordinary
            || substitute.governance != DependencyGovernance::Ordinary
        {
            return Err(RegenerativeSubstitutionError::SafeguardedSubstitutionForbidden);
        }
        if baseline.kind != substitute.kind {
            return Err(RegenerativeSubstitutionError::DependencyKindMismatch);
        }

        let mut derived = source.clone();
        let derived_capability = derived
            .capabilities
            .iter_mut()
            .find(|capability| capability.capability_id == self.capability_id)
            .ok_or(RegenerativeSubstitutionError::CapabilityNotFound)?;
        if !derived_capability
            .dependency_ids
            .remove(&self.baseline_dependency_id)
        {
            return Err(RegenerativeSubstitutionError::BaselineNotRequired);
        }
        derived_capability
            .dependency_ids
            .insert(self.substitute_dependency_id.clone());
        derived.model_id = self.derived_model_id.clone();
        derived.evidence_binding = self.derived_model_evidence_binding.clone();
        derived
            .validate()
            .map_err(|_| RegenerativeSubstitutionError::InvalidDerivedModel)?;
        Ok(derived)
    }
}

/// Structural refusal reasons for regenerative substitution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeSubstitutionError {
    /// Schema version is not understood by this implementation.
    UnsupportedSchema { schema_version: u8 },
    /// One identifier is empty, padded, oversized, or control-bearing.
    InvalidIdentifier,
    /// One evidence binding is empty, padded, oversized, or control-bearing.
    InvalidEvidenceBinding,
    /// Baseline and substitute name the same dependency.
    IdentitySubstitution,
    /// Derived model must receive a distinct identifier and evidence binding.
    DerivedModelNotDistinct,
    /// Source model failed its own closure-model validation.
    InvalidSourceModel,
    /// Source model identifier or evidence binding does not match the claim.
    SourceModelMismatch,
    /// Named capability is absent from the source model.
    CapabilityNotFound,
    /// Capability evidence binding does not match the source model.
    CapabilityBindingMismatch,
    /// Baseline dependency is not currently required by the capability.
    BaselineNotRequired,
    /// Substitute is already independently required by the capability.
    SubstituteAlreadyRequired,
    /// Baseline dependency is absent from the source model.
    BaselineDependencyNotFound,
    /// Substitute dependency is absent from the source model.
    SubstituteDependencyNotFound,
    /// Generic v1 substitution cannot cross a safeguarded-external boundary.
    SafeguardedSubstitutionForbidden,
    /// Baseline and substitute have different dependency kinds.
    DependencyKindMismatch,
    /// Derived closure model failed validation after substitution.
    InvalidDerivedModel,
}

fn validate_id(value: &str) -> Result<(), RegenerativeSubstitutionError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeSubstitutionError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeSubstitutionError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeSubstitutionError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RegenerativeCapability, RegenerativeDependency, RegenerativeDependencyKind,
        RegenerativeHorizon,
    };
    use std::collections::BTreeSet;

    fn dependency(
        id: &str,
        governance: DependencyGovernance,
        stockpile: u64,
        production: u64,
    ) -> RegenerativeDependency {
        RegenerativeDependency {
            dependency_id: id.into(),
            kind: RegenerativeDependencyKind::Component,
            governance,
            demand_units_per_period: 1,
            local_production_units_per_period: production,
            recycling_units_per_period: 0,
            stockpile_units: stockpile,
            unit_mass_grams: Some(1_000),
            evidence_binding: format!("evidence:{id}"),
        }
    }

    fn model() -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: "manta-baseline".into(),
            period_duration_ms: 1,
            dependencies: vec![
                dependency("imported-motor", DependencyGovernance::Ordinary, 5, 0),
                dependency("local-motor", DependencyGovernance::Ordinary, 0, 1),
                dependency(
                    "qualified-reactor-service",
                    DependencyGovernance::SafeguardedExternal,
                    30,
                    0,
                ),
            ],
            capabilities: vec![RegenerativeCapability {
                capability_id: "pump-service".into(),
                essential: true,
                dependency_ids: BTreeSet::from(["imported-motor".into()]),
                evidence_binding: "evidence:pump-service".into(),
            }],
            evidence_binding: "evidence:manta-baseline".into(),
        }
    }

    fn substitution() -> RegenerativeSubstitutionEvidenceV1 {
        RegenerativeSubstitutionEvidenceV1 {
            schema_version: REGENERATIVE_SUBSTITUTION_SCHEMA_V1,
            source_model_id: "manta-baseline".into(),
            source_model_evidence_binding: "evidence:manta-baseline".into(),
            derived_model_id: "manta-local-motor-scenario".into(),
            derived_model_evidence_binding: "evidence:manta-local-motor-scenario".into(),
            capability_id: "pump-service".into(),
            capability_evidence_binding: "evidence:pump-service".into(),
            baseline_dependency_id: "imported-motor".into(),
            substitute_dependency_id: "local-motor".into(),
            qualification_binding: "qualification:local-motor-for-pump-service".into(),
        }
    }

    #[test]
    fn explicit_ordinary_substitution_can_extend_a_diagnostic_horizon() {
        let source = model();
        assert_eq!(
            source.evaluate().unwrap().essential_horizon,
            RegenerativeHorizon::FinitePeriods(5)
        );
        let derived = substitution().apply_to_model(&source).unwrap();
        assert_eq!(
            derived.evaluate().unwrap().essential_horizon,
            RegenerativeHorizon::IndefiniteUnderStaticModel
        );
        assert_eq!(source.model_id, "manta-baseline");
        assert!(source.capabilities[0]
            .dependency_ids
            .contains("imported-motor"));
    }

    #[test]
    fn safeguarded_dependencies_cannot_use_generic_substitution() {
        let mut source = model();
        source.capabilities[0].dependency_ids =
            BTreeSet::from(["qualified-reactor-service".into()]);
        let mut claim = substitution();
        claim.baseline_dependency_id = "qualified-reactor-service".into();
        assert_eq!(
            claim.apply_to_model(&source),
            Err(RegenerativeSubstitutionError::SafeguardedSubstitutionForbidden)
        );
    }

    #[test]
    fn source_and_capability_bindings_are_exact() {
        let source = model();
        let mut wrong_model = substitution();
        wrong_model.source_model_evidence_binding = "evidence:other".into();
        assert_eq!(
            wrong_model.apply_to_model(&source),
            Err(RegenerativeSubstitutionError::SourceModelMismatch)
        );

        let mut wrong_capability = substitution();
        wrong_capability.capability_evidence_binding = "evidence:other".into();
        assert_eq!(
            wrong_capability.apply_to_model(&source),
            Err(RegenerativeSubstitutionError::CapabilityBindingMismatch)
        );
    }

    #[test]
    fn duplicate_requirement_collapse_is_rejected() {
        let mut source = model();
        source.capabilities[0]
            .dependency_ids
            .insert("local-motor".into());
        assert_eq!(
            substitution().apply_to_model(&source),
            Err(RegenerativeSubstitutionError::SubstituteAlreadyRequired)
        );
    }
}
