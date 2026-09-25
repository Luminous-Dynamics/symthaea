// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bound lifecycle implications for manufacturing processes.
//!
//! This crate does not calculate environmental impact, prove repairability, or authorize
//! manufacturing. It binds exact process/lifecycle subjects to non-scalar design relations.
//!
//! ```text
//! process has lifecycle relation
//! != lifecycle outcome improved
//! != environmental benefit established
//! != recovery route feasible
//! != circularity certified
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_lifecycle_engineering::ExactLifecycleSubjectRefV1;
use symthaea_manufacturing_process::ProcessDefinitionId;
use thiserror::Error;

const BRIDGE_DOMAIN: &str = "symthaea-manufacturing-lifecycle-bridge::v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManufacturingLifecycleBridgeError {
    #[error("{field} must be a canonical non-empty token")]
    NonCanonical { field: &'static str },
    #[error("bridge requires at least one lifecycle relation")]
    MissingRelation,
    #[error("duplicate lifecycle relation: {0}")]
    DuplicateRelation(String),
    #[error("invalid external lifecycle reference: {0}")]
    InvalidExternalRef(String),
}

fn canonical(field: &'static str, value: &str) -> Result<(), ManufacturingLifecycleBridgeError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ManufacturingLifecycleBridgeError::NonCanonical { field });
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessLifecycleRelationKindV1 {
    RequiresEnvironmentalInventory,
    RequiresEnergyAssessment,
    RequiresWaterAssessment,
    RequiresConsumablesAssessment,
    RequiresCriticalMaterialAssessment,
    RequiresHazardousEndOfLifeHandling,
    ConstrainsDisassemblyRoute,
    ConstrainsMaterialSeparation,
    MayComplicateRecoveryRoute,
    EnablesCandidateRepairRoute,
    EnablesCandidateRemanufactureRoute,
    EnablesCandidateComponentHarvestRoute,
    ProducesPotentialSecondaryMaterialStream,
    RequiresRecoveryProcessAssessment,
    RequiresResidualWasteDisposition,
    UnknownOrAssessmentRequired,
}

impl ProcessLifecycleRelationKindV1 {
    fn tag(self) -> &'static str {
        match self {
            Self::RequiresEnvironmentalInventory => "requires-environmental-inventory",
            Self::RequiresEnergyAssessment => "requires-energy-assessment",
            Self::RequiresWaterAssessment => "requires-water-assessment",
            Self::RequiresConsumablesAssessment => "requires-consumables-assessment",
            Self::RequiresCriticalMaterialAssessment => "requires-critical-material-assessment",
            Self::RequiresHazardousEndOfLifeHandling => "requires-hazardous-end-of-life-handling",
            Self::ConstrainsDisassemblyRoute => "constrains-disassembly-route",
            Self::ConstrainsMaterialSeparation => "constrains-material-separation",
            Self::MayComplicateRecoveryRoute => "may-complicate-recovery-route",
            Self::EnablesCandidateRepairRoute => "enables-candidate-repair-route",
            Self::EnablesCandidateRemanufactureRoute => "enables-candidate-remanufacture-route",
            Self::EnablesCandidateComponentHarvestRoute => "enables-candidate-component-harvest-route",
            Self::ProducesPotentialSecondaryMaterialStream => "produces-potential-secondary-material-stream",
            Self::RequiresRecoveryProcessAssessment => "requires-recovery-process-assessment",
            Self::RequiresResidualWasteDisposition => "requires-residual-waste-disposition",
            Self::UnknownOrAssessmentRequired => "unknown-or-assessment-required",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessLifecycleRelationV1 {
    pub kind: ProcessLifecycleRelationKindV1,
    /// Exact evidence, assessment, strategy, or route subject that justifies/defines the relation.
    pub subject: ExactLifecycleSubjectRefV1,
}

impl ProcessLifecycleRelationV1 {
    pub fn validate(&self) -> Result<(), ManufacturingLifecycleBridgeError> {
        self.subject
            .validate()
            .map_err(|err| ManufacturingLifecycleBridgeError::InvalidExternalRef(err.to_string()))
    }

    fn identity_key(&self) -> Result<String, ManufacturingLifecycleBridgeError> {
        self.validate()?;
        Ok(format!("{}:{}", self.kind.tag(), self.subject.ref_id().map_err(|err| {
            ManufacturingLifecycleBridgeError::InvalidExternalRef(err.to_string())
        })?))
    }
}

/// Maximum authority of this bridge. It expresses design relations and assessment requirements only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManufacturingLifecycleAuthorityCeilingV1 {
    DesignRelationsAndAssessmentRequirementsOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManufacturingLifecycleBridgeV1 {
    pub semantic_version: String,
    pub process_id: ProcessDefinitionId,
    /// Optional exact process-pack/profile subject when the process came from an extension pack.
    pub process_pack_profile: Option<ExactLifecycleSubjectRefV1>,
    /// Exact lifecycle-design-profile subject from `symthaea-lifecycle-engineering`.
    pub lifecycle_design_profile: ExactLifecycleSubjectRefV1,
    #[serde(default)]
    pub relations: Vec<ProcessLifecycleRelationV1>,
    pub authority_ceiling: ManufacturingLifecycleAuthorityCeilingV1,
    pub display_label: Option<String>,
    pub notes: Option<String>,
}

impl ManufacturingLifecycleBridgeV1 {
    pub fn validate(&self) -> Result<(), ManufacturingLifecycleBridgeError> {
        canonical("bridge.semantic_version", &self.semantic_version)?;
        canonical("bridge.process_id", &self.process_id.0)?;
        if let Some(reference) = &self.process_pack_profile {
            reference.validate().map_err(|err| {
                ManufacturingLifecycleBridgeError::InvalidExternalRef(err.to_string())
            })?;
        }
        self.lifecycle_design_profile.validate().map_err(|err| {
            ManufacturingLifecycleBridgeError::InvalidExternalRef(err.to_string())
        })?;
        if self.relations.is_empty() {
            return Err(ManufacturingLifecycleBridgeError::MissingRelation);
        }
        let mut seen = BTreeSet::new();
        for relation in &self.relations {
            let key = relation.identity_key()?;
            if !seen.insert(key.clone()) {
                return Err(ManufacturingLifecycleBridgeError::DuplicateRelation(key));
            }
        }
        if let Some(label) = &self.display_label {
            canonical("bridge.display_label", label)?;
        }
        if let Some(notes) = &self.notes {
            canonical("bridge.notes", notes)?;
        }
        Ok(())
    }

    pub fn bridge_id(&self) -> Result<String, ManufacturingLifecycleBridgeError> {
        self.validate()?;
        let mut relations = self.relations.clone();
        relations.sort_by_key(|relation| relation.kind);
        let mut keyed = relations
            .iter()
            .map(ProcessLifecycleRelationV1::identity_key)
            .collect::<Result<Vec<_>, _>>()?;
        keyed.sort();

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, BRIDGE_DOMAIN);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(&mut hasher, &self.process_id.0);
        if let Some(reference) = &self.process_pack_profile {
            hash_field(
                &mut hasher,
                &reference.ref_id().map_err(|err| {
                    ManufacturingLifecycleBridgeError::InvalidExternalRef(err.to_string())
                })?,
            );
        } else {
            hash_field(&mut hasher, "no-process-pack-profile");
        }
        hash_field(
            &mut hasher,
            &self.lifecycle_design_profile.ref_id().map_err(|err| {
                ManufacturingLifecycleBridgeError::InvalidExternalRef(err.to_string())
            })?,
        );
        for key in keyed {
            hash_field(&mut hasher, &key);
        }
        hash_field(
            &mut hasher,
            "authority:design-relations-and-assessment-requirements-only",
        );
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn subject(namespace: &str, id: &str, fill: char) -> ExactLifecycleSubjectRefV1 {
        ExactLifecycleSubjectRefV1 {
            namespace: namespace.into(),
            subject_id: id.into(),
            semantic_version: "1".into(),
            content_blake3: digest(fill),
        }
    }

    fn relation(kind: ProcessLifecycleRelationKindV1, id: &str, fill: char) -> ProcessLifecycleRelationV1 {
        ProcessLifecycleRelationV1 {
            kind,
            subject: subject("lifecycle.assessment", id, fill),
        }
    }

    fn bridge() -> ManufacturingLifecycleBridgeV1 {
        ManufacturingLifecycleBridgeV1 {
            semantic_version: "1.0.0".into(),
            process_id: ProcessDefinitionId("process:cold-spray-repair-v1".into()),
            process_pack_profile: Some(subject("mfg.pack", "advanced:cold-spray", 'a')),
            lifecycle_design_profile: subject("symthaea.lifecycle", "profile-1", 'b'),
            relations: vec![
                relation(ProcessLifecycleRelationKindV1::EnablesCandidateRepairRoute, "repair-route", 'c'),
                relation(ProcessLifecycleRelationKindV1::RequiresEnergyAssessment, "energy-inventory", 'd'),
            ],
            authority_ceiling: ManufacturingLifecycleAuthorityCeilingV1::DesignRelationsAndAssessmentRequirementsOnly,
            display_label: Some("Cold spray lifecycle bridge".into()),
            notes: Some("Design relation only".into()),
        }
    }

    #[test]
    fn display_metadata_does_not_change_bridge_identity() {
        let a = bridge();
        let mut b = a.clone();
        b.display_label = Some("Renamed".into());
        b.notes = Some("Different navigation prose".into());
        assert_eq!(a.bridge_id().unwrap(), b.bridge_id().unwrap());
    }

    #[test]
    fn relation_change_changes_bridge_identity() {
        let a = bridge();
        let mut b = a.clone();
        b.relations.push(relation(
            ProcessLifecycleRelationKindV1::RequiresWaterAssessment,
            "water-inventory",
            'e',
        ));
        assert_ne!(a.bridge_id().unwrap(), b.bridge_id().unwrap());
    }

    #[test]
    fn candidate_benefit_and_constraint_can_coexist() {
        let mut value = bridge();
        value.relations.push(relation(
            ProcessLifecycleRelationKindV1::MayComplicateRecoveryRoute,
            "recovery-route-assessment",
            'f',
        ));
        assert!(value.validate().is_ok());
    }

    #[test]
    fn exotic_unknown_can_remain_explicitly_unresolved() {
        let value = ManufacturingLifecycleBridgeV1 {
            semantic_version: "1.0.0".into(),
            process_id: ProcessDefinitionId("process:containerless-research-v1".into()),
            process_pack_profile: Some(subject("mfg.pack", "research:containerless", '1')),
            lifecycle_design_profile: subject("symthaea.lifecycle", "research-profile", '2'),
            relations: vec![relation(
                ProcessLifecycleRelationKindV1::UnknownOrAssessmentRequired,
                "lifecycle-evidence-needed",
                '3',
            )],
            authority_ceiling: ManufacturingLifecycleAuthorityCeilingV1::DesignRelationsAndAssessmentRequirementsOnly,
            display_label: None,
            notes: None,
        };
        assert!(value.validate().is_ok());
    }

    #[test]
    fn duplicate_relation_rejects() {
        let mut value = bridge();
        value.relations.push(value.relations[0].clone());
        assert!(matches!(
            value.validate(),
            Err(ManufacturingLifecycleBridgeError::DuplicateRelation(_))
        ));
    }

    #[test]
    fn serialization_contains_no_scalar_sustainability_score() {
        let encoded = serde_json::to_string(&bridge()).unwrap();
        assert!(!encoded.contains("sustainability_score"));
        let decoded: ManufacturingLifecycleBridgeV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.bridge_id(), bridge().bridge_id());
    }
}
