// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Portable exact-subject projection for lifecycle engineering contracts.
//!
//! This crate is an interoperability adapter only. It reuses Symthaea's existing
//! `ExactLifecycleSubjectRefV1` envelope rather than defining another identity ontology.
//!
//! ```text
//! canonical Symthaea subject
//! -> portable exact tuple
//! != external authority resolution
//! != currentness
//! != execution evidence
//! ```

use symthaea_lifecycle_engineering::{ExactLifecycleSubjectRefV1, LifecycleDesignProfileV1};
use symthaea_manufacturing_lifecycle_bridge::ManufacturingLifecycleBridgeV1;
use thiserror::Error;

pub const LIFECYCLE_PROFILE_NAMESPACE_V1: &str = "symthaea.lifecycle-design.profile";
pub const MANUFACTURING_LIFECYCLE_BRIDGE_NAMESPACE_V1: &str =
    "symthaea.manufacturing-lifecycle.bridge";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LifecyclePortabilityError {
    #[error("invalid lifecycle design subject: {0}")]
    LifecycleDesign(String),
    #[error("invalid manufacturing lifecycle bridge subject: {0}")]
    ManufacturingLifecycleBridge(String),
}

/// Project a canonical Symthaea subject into a portable exact-subject tuple.
///
/// Consumers should compare/preserve the tuple fields themselves across repository
/// boundaries. `ExactLifecycleSubjectRefV1::ref_id()` remains a Symthaea-local
/// domain-separated wrapper identity and is not the cross-repository equality key.
pub trait PortableLifecycleSubjectV1 {
    fn portable_exact_ref(&self) -> Result<ExactLifecycleSubjectRefV1, LifecyclePortabilityError>;
}

impl PortableLifecycleSubjectV1 for LifecycleDesignProfileV1 {
    fn portable_exact_ref(&self) -> Result<ExactLifecycleSubjectRefV1, LifecyclePortabilityError> {
        let id = self
            .profile_id()
            .map_err(|err| LifecyclePortabilityError::LifecycleDesign(err.to_string()))?;
        let reference = ExactLifecycleSubjectRefV1 {
            namespace: LIFECYCLE_PROFILE_NAMESPACE_V1.into(),
            subject_id: id.clone(),
            semantic_version: self.semantic_version.clone(),
            content_blake3: id,
        };
        reference
            .validate()
            .map_err(|err| LifecyclePortabilityError::LifecycleDesign(err.to_string()))?;
        Ok(reference)
    }
}

impl PortableLifecycleSubjectV1 for ManufacturingLifecycleBridgeV1 {
    fn portable_exact_ref(&self) -> Result<ExactLifecycleSubjectRefV1, LifecyclePortabilityError> {
        let id = self.bridge_id().map_err(|err| {
            LifecyclePortabilityError::ManufacturingLifecycleBridge(err.to_string())
        })?;
        let reference = ExactLifecycleSubjectRefV1 {
            namespace: MANUFACTURING_LIFECYCLE_BRIDGE_NAMESPACE_V1.into(),
            subject_id: id.clone(),
            semantic_version: self.semantic_version.clone(),
            content_blake3: id,
        };
        reference.validate().map_err(|err| {
            LifecyclePortabilityError::ManufacturingLifecycleBridge(err.to_string())
        })?;
        Ok(reference)
    }
}

/// Explicitly states the authority ceiling of this adapter.
/// A portable reference preserves identity fields only; it does not resolve or authenticate them.
pub fn claims_external_authority_resolution() -> bool {
    false
}

pub fn claims_subject_currentness() -> bool {
    false
}

pub fn claims_execution_evidence() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_lifecycle_engineering::{
        LifecycleAssessmentRefV1, LifecycleDesignAuthorityCeilingV1, LifecycleStrategyKindV1,
        LifecycleStrategyRefV1,
    };
    use symthaea_manufacturing_lifecycle_bridge::{
        ManufacturingLifecycleAuthorityCeilingV1, ProcessLifecycleRelationKindV1,
        ProcessLifecycleRelationV1,
    };
    use symthaea_manufacturing_process::ProcessDefinitionId;

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

    fn profile() -> LifecycleDesignProfileV1 {
        LifecycleDesignProfileV1 {
            semantic_version: "1.0.0".into(),
            subject_configuration: subject("symthaea.configuration", "speaker-v1", 'a'),
            functional_unit: subject("symthaea.functional-unit", "one-serviceable-speaker", 'b'),
            manufacturing_plan_refs: vec![subject("symthaea.mfg-plan", "plan-v1", 'c')],
            strategies: vec![LifecycleStrategyRefV1 {
                kind: LifecycleStrategyKindV1::Repair,
                subject: subject("symthaea.lifecycle.strategy", "repair-v1", 'd'),
            }],
            assessments: Vec::<LifecycleAssessmentRefV1>::new(),
            authority_ceiling: LifecycleDesignAuthorityCeilingV1::DesignIntentAndAssessmentReferencesOnly,
            display_label: Some("Serviceable speaker lifecycle".into()),
            notes: Some("portable-ref fixture".into()),
        }
    }

    fn bridge(profile_ref: ExactLifecycleSubjectRefV1) -> ManufacturingLifecycleBridgeV1 {
        ManufacturingLifecycleBridgeV1 {
            semantic_version: "1.0.0".into(),
            process_id: ProcessDefinitionId("process:cold-spray-repair-v1".into()),
            process_pack_profile: None,
            lifecycle_design_profile: profile_ref,
            relations: vec![ProcessLifecycleRelationV1 {
                kind: ProcessLifecycleRelationKindV1::EnablesCandidateRepairRoute,
                subject: subject("symthaea.lifecycle.assessment", "repair-route-evidence", 'e'),
            }],
            authority_ceiling:
                ManufacturingLifecycleAuthorityCeilingV1::DesignRelationsAndAssessmentRequirementsOnly,
            display_label: Some("Repair bridge".into()),
            notes: Some("portable-ref fixture".into()),
        }
    }

    #[test]
    fn lifecycle_profile_exports_frozen_portable_namespace() {
        let reference = profile().portable_exact_ref().unwrap();
        assert_eq!(reference.namespace, LIFECYCLE_PROFILE_NAMESPACE_V1);
        assert_eq!(reference.subject_id, reference.content_blake3);
        assert_eq!(reference.semantic_version, "1.0.0");
        assert!(reference.validate().is_ok());
    }

    #[test]
    fn bridge_exports_frozen_portable_namespace() {
        let profile_ref = profile().portable_exact_ref().unwrap();
        let reference = bridge(profile_ref).portable_exact_ref().unwrap();
        assert_eq!(
            reference.namespace,
            MANUFACTURING_LIFECYCLE_BRIDGE_NAMESPACE_V1
        );
        assert_eq!(reference.subject_id, reference.content_blake3);
        assert_eq!(reference.semantic_version, "1.0.0");
        assert!(reference.validate().is_ok());
    }

    #[test]
    fn display_metadata_does_not_change_portable_profile_tuple() {
        let a = profile();
        let mut b = a.clone();
        b.display_label = Some("Renamed UI label".into());
        b.notes = Some("Different navigation prose".into());
        assert_eq!(
            a.portable_exact_ref().unwrap(),
            b.portable_exact_ref().unwrap()
        );
    }

    #[test]
    fn display_metadata_does_not_change_portable_bridge_tuple() {
        let profile_ref = profile().portable_exact_ref().unwrap();
        let a = bridge(profile_ref);
        let mut b = a.clone();
        b.display_label = Some("Renamed UI label".into());
        b.notes = Some("Different navigation prose".into());
        assert_eq!(
            a.portable_exact_ref().unwrap(),
            b.portable_exact_ref().unwrap()
        );
    }

    #[test]
    fn semantic_relation_change_changes_portable_bridge_tuple() {
        let profile_ref = profile().portable_exact_ref().unwrap();
        let a = bridge(profile_ref.clone());
        let mut b = bridge(profile_ref);
        b.relations.push(ProcessLifecycleRelationV1 {
            kind: ProcessLifecycleRelationKindV1::RequiresEnergyAssessment,
            subject: subject("symthaea.lifecycle.assessment", "energy-v1", 'f'),
        });
        assert_ne!(
            a.portable_exact_ref().unwrap(),
            b.portable_exact_ref().unwrap()
        );
    }

    #[test]
    fn serde_round_trip_preserves_raw_portable_tuple() {
        let reference = profile().portable_exact_ref().unwrap();
        let encoded = serde_json::to_string(&reference).unwrap();
        let decoded: ExactLifecycleSubjectRefV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(reference, decoded);
    }

    #[test]
    fn portability_adapter_claims_no_external_authority_or_execution() {
        assert!(!claims_external_authority_resolution());
        assert!(!claims_subject_currentness());
        assert!(!claims_execution_evidence());
    }
}
