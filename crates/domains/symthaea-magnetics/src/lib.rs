// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable magnetic assembly, coupling and evidence semantics.
//!
//! This crate does not own canonical physical quantities, component catalog identity,
//! material truth, numerical EM solvers, FIELD observations, or physical actuation.
//! It defines the exact magnetic assembly/model subject those systems may refer to.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_engineering_catalog::ComponentSubjectId;
use thiserror::Error;

const ASSEMBLY_ID_DOMAIN: &str = "symthaea-magnetics::assembly-v1";
const MODEL_ID_DOMAIN: &str = "symthaea-magnetics::model-profile-v1";
const SNAPSHOT_ID_DOMAIN: &str = "symthaea-magnetics::evidence-snapshot-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MagneticError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("{field} contains leading or trailing whitespace")]
    NonCanonicalWhitespace { field: &'static str },
    #[error("duplicate magnetic member id: {0}")]
    DuplicateMember(String),
    #[error("duplicate magnetic relation id: {0}")]
    DuplicateRelation(String),
    #[error("magnetic relation references unknown member: {0}")]
    UnknownMember(String),
    #[error("magnetic relation cannot self-reference: {0}")]
    SelfRelation(String),
    #[error("member role requires a catalog component subject: {0}")]
    ComponentRequired(String),
    #[error("field-region role must not carry a catalog component subject: {0}")]
    FieldRegionCannotBeCatalogComponent(String),
    #[error("model profile is bound to a different assembly")]
    AssemblyMismatch,
    #[error("model profile requires source/model evidence")]
    PredictiveEvidenceRequired,
    #[error("unknown/not-admitted model family cannot claim model authority")]
    UnknownModelNotPredictive,
    #[error("evidence reference points at a different assembly")]
    EvidenceAssemblyMismatch,
    #[error("evidence plane mismatch")]
    EvidencePlaneMismatch,
    #[error("duplicate magnetic model-profile identity: {0}")]
    DuplicateModelProfile(String),
    #[error("evidence references a model profile absent from the snapshot: {0}")]
    UnknownModelProfile(String),
    #[error("duplicate magnetic evidence identity: {0}")]
    DuplicateEvidence(String),
    #[error("predicted magnetic evidence requires an exact local model-profile reference")]
    PredictedEvidenceRequiresModelProfile,
    #[error("external FEM requirement cannot itself back magnetic prediction evidence")]
    RequirementMarkerCannotBackPrediction,
    #[error("imported predicted field map requires result-receipt closure before admitted prediction evidence")]
    ImportedPredictionNotClosed,
    #[error("observed/measured magnetic evidence must not use a model profile as measurement authority")]
    MeasuredEvidenceCannotUseModelProfile,
}

fn validate_token(field: &'static str, value: &str) -> Result<(), MagneticError> {
    if value.is_empty() {
        return Err(MagneticError::EmptyField { field });
    }
    if value.trim() != value {
        return Err(MagneticError::NonCanonicalWhitespace { field });
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MagneticAssemblyId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MagneticModelProfileId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MagneticEvidenceSnapshotId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MagneticMemberRoleV1 {
    PermanentMagnet,
    ExcitationWinding,
    MagneticCore,
    PolePieceOrYoke,
    AirGap,
    FluxGuide,
    Shield,
    MagneticSensor,
    NonMagneticSpacer,
    ThermalInterface,
    MechanicalInterface,
    FieldRegion,
    Other(String),
}

impl MagneticMemberRoleV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::PermanentMagnet => "permanent-magnet".into(),
            Self::ExcitationWinding => "excitation-winding".into(),
            Self::MagneticCore => "magnetic-core".into(),
            Self::PolePieceOrYoke => "pole-piece-or-yoke".into(),
            Self::AirGap => "air-gap".into(),
            Self::FluxGuide => "flux-guide".into(),
            Self::Shield => "shield".into(),
            Self::MagneticSensor => "magnetic-sensor".into(),
            Self::NonMagneticSpacer => "non-magnetic-spacer".into(),
            Self::ThermalInterface => "thermal-interface".into(),
            Self::MechanicalInterface => "mechanical-interface".into(),
            Self::FieldRegion => "field-region".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), MagneticError> {
        if let Self::Other(value) = self {
            validate_token("member_role.other", value)?;
        }
        Ok(())
    }

    fn component_required(&self) -> bool {
        matches!(
            self,
            Self::PermanentMagnet
                | Self::ExcitationWinding
                | Self::MagneticCore
                | Self::PolePieceOrYoke
                | Self::FluxGuide
                | Self::Shield
                | Self::MagneticSensor
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagneticMemberV1 {
    pub member_id: String,
    pub role: MagneticMemberRoleV1,
    pub component_subject_id: Option<ComponentSubjectId>,
    pub geometry_binding_id: Option<String>,
    pub material_state_ref: Option<String>,
    pub orientation_profile_id: Option<String>,
    pub display_name: Option<String>,
}

impl MagneticMemberV1 {
    pub fn validate(&self) -> Result<(), MagneticError> {
        validate_token("member_id", &self.member_id)?;
        self.role.validate()?;
        if self.role.component_required() && self.component_subject_id.is_none() {
            return Err(MagneticError::ComponentRequired(self.member_id.clone()));
        }
        if self.role == MagneticMemberRoleV1::FieldRegion && self.component_subject_id.is_some() {
            return Err(MagneticError::FieldRegionCannotBeCatalogComponent(
                self.member_id.clone(),
            ));
        }
        for (field, value) in [
            ("geometry_binding_id", self.geometry_binding_id.as_deref()),
            ("material_state_ref", self.material_state_ref.as_deref()),
            ("orientation_profile_id", self.orientation_profile_id.as_deref()),
        ] {
            if let Some(value) = value {
                validate_token(field, value)?;
            }
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}",
            self.member_id,
            self.role.canonical_tag(),
            self.component_subject_id
                .as_ref()
                .map(|id| id.0.as_str())
                .unwrap_or("none"),
            self.geometry_binding_id.as_deref().unwrap_or("none"),
            self.material_state_ref.as_deref().unwrap_or("none"),
            self.orientation_profile_id.as_deref().unwrap_or("none")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MagneticRelationKindV1 {
    FluxPathConnection,
    AirGapInterface,
    WindingExcitesRegion,
    PermanentMagnetBiasesRegion,
    SensorObservesRegion,
    MechanicalLoadCouplingRef,
    ThermalCouplingRef,
    GeometryBindingRef,
    Other(String),
}

impl MagneticRelationKindV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::FluxPathConnection => "flux-path-connection".into(),
            Self::AirGapInterface => "air-gap-interface".into(),
            Self::WindingExcitesRegion => "winding-excites-region".into(),
            Self::PermanentMagnetBiasesRegion => "permanent-magnet-biases-region".into(),
            Self::SensorObservesRegion => "sensor-observes-region".into(),
            Self::MechanicalLoadCouplingRef => "mechanical-load-coupling-ref".into(),
            Self::ThermalCouplingRef => "thermal-coupling-ref".into(),
            Self::GeometryBindingRef => "geometry-binding-ref".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), MagneticError> {
        if let Self::Other(value) = self {
            validate_token("relation_kind.other", value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagneticRelationV1 {
    pub relation_id: String,
    pub kind: MagneticRelationKindV1,
    pub from_member_id: String,
    pub to_member_id: String,
    pub coupling_profile_id: Option<String>,
}

impl MagneticRelationV1 {
    pub fn validate(&self) -> Result<(), MagneticError> {
        validate_token("relation_id", &self.relation_id)?;
        validate_token("from_member_id", &self.from_member_id)?;
        validate_token("to_member_id", &self.to_member_id)?;
        self.kind.validate()?;
        if let Some(value) = self.coupling_profile_id.as_deref() {
            validate_token("coupling_profile_id", value)?;
        }
        if self.from_member_id == self.to_member_id {
            return Err(MagneticError::SelfRelation(self.relation_id.clone()));
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.relation_id,
            self.kind.canonical_tag(),
            self.from_member_id,
            self.to_member_id,
            self.coupling_profile_id.as_deref().unwrap_or("none")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagneticAssemblyV1 {
    pub assembly_kind_id: String,
    #[serde(default)]
    pub members: Vec<MagneticMemberV1>,
    #[serde(default)]
    pub relations: Vec<MagneticRelationV1>,
    pub display_name: Option<String>,
}

impl MagneticAssemblyV1 {
    pub fn validate(&self) -> Result<(), MagneticError> {
        validate_token("assembly_kind_id", &self.assembly_kind_id)?;
        let mut members = BTreeSet::new();
        for member in &self.members {
            member.validate()?;
            if !members.insert(member.member_id.clone()) {
                return Err(MagneticError::DuplicateMember(member.member_id.clone()));
            }
        }
        let mut relations = BTreeSet::new();
        for relation in &self.relations {
            relation.validate()?;
            if !relations.insert(relation.relation_id.clone()) {
                return Err(MagneticError::DuplicateRelation(relation.relation_id.clone()));
            }
            if !members.contains(&relation.from_member_id) {
                return Err(MagneticError::UnknownMember(relation.from_member_id.clone()));
            }
            if !members.contains(&relation.to_member_id) {
                return Err(MagneticError::UnknownMember(relation.to_member_id.clone()));
            }
        }
        Ok(())
    }

    pub fn assembly_id(&self) -> Result<MagneticAssemblyId, MagneticError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, ASSEMBLY_ID_DOMAIN);
        hash_field(&mut hasher, &self.assembly_kind_id);
        let mut members = self
            .members
            .iter()
            .map(MagneticMemberV1::canonical_key)
            .collect::<Vec<_>>();
        members.sort();
        for member in members {
            hash_field(&mut hasher, &member);
        }
        let mut relations = self
            .relations
            .iter()
            .map(MagneticRelationV1::canonical_key)
            .collect::<Vec<_>>();
        relations.sort();
        for relation in relations {
            hash_field(&mut hasher, &relation);
        }
        Ok(MagneticAssemblyId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MagneticModelFamilyV1 {
    LinearReluctance,
    NonlinearBhCurve,
    Hysteresis,
    PermanentMagnetRecoilDemagnetization,
    EddyCurrentFrequencyDependent,
    ImportedPredictedFieldMap,
    ImportedMeasuredFieldMap,
    ExternalFiniteElementRequired,
    UnknownNotAdmitted,
    Other(String),
}

impl MagneticModelFamilyV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::LinearReluctance => "linear-reluctance".into(),
            Self::NonlinearBhCurve => "nonlinear-bh-curve".into(),
            Self::Hysteresis => "hysteresis".into(),
            Self::PermanentMagnetRecoilDemagnetization => {
                "permanent-magnet-recoil-demagnetization".into()
            }
            Self::EddyCurrentFrequencyDependent => "eddy-current-frequency-dependent".into(),
            Self::ImportedPredictedFieldMap => "imported-predicted-field-map".into(),
            Self::ImportedMeasuredFieldMap => "imported-measured-field-map".into(),
            Self::ExternalFiniteElementRequired => "external-finite-element-required".into(),
            Self::UnknownNotAdmitted => "unknown-not-admitted".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), MagneticError> {
        if let Self::Other(value) = self {
            validate_token("model_family.other", value)?;
        }
        Ok(())
    }

    pub fn is_backend_requirement(&self) -> bool {
        matches!(self, Self::ExternalFiniteElementRequired)
    }

    pub fn supports_prediction(&self) -> bool {
        !matches!(
            self,
            Self::UnknownNotAdmitted
                | Self::ImportedPredictedFieldMap
                | Self::ImportedMeasuredFieldMap
                | Self::ExternalFiniteElementRequired
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagneticModelProfileV1 {
    pub assembly_id: MagneticAssemblyId,
    pub model_family: MagneticModelFamilyV1,
    pub applicability_profile_id: String,
    #[serde(default)]
    pub source_evidence_refs: Vec<String>,
    #[serde(default)]
    pub geometry_binding_refs: Vec<String>,
    #[serde(default)]
    pub material_evidence_refs: Vec<String>,
}

impl MagneticModelProfileV1 {
    pub fn validate_for(&self, assembly: &MagneticAssemblyV1) -> Result<(), MagneticError> {
        self.model_family.validate()?;
        validate_token("applicability_profile_id", &self.applicability_profile_id)?;
        if self.assembly_id != assembly.assembly_id()? {
            return Err(MagneticError::AssemblyMismatch);
        }
        if self.model_family == MagneticModelFamilyV1::UnknownNotAdmitted {
            return Err(MagneticError::UnknownModelNotPredictive);
        }
        for (field, values) in [
            ("source_evidence_ref", &self.source_evidence_refs),
            ("geometry_binding_ref", &self.geometry_binding_refs),
            ("material_evidence_ref", &self.material_evidence_refs),
        ] {
            for value in values {
                validate_token(field, value)?;
            }
        }
        if self.source_evidence_refs.is_empty() {
            return Err(MagneticError::PredictiveEvidenceRequired);
        }
        Ok(())
    }

    pub fn model_profile_id(
        &self,
        assembly: &MagneticAssemblyV1,
    ) -> Result<MagneticModelProfileId, MagneticError> {
        self.validate_for(assembly)?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, MODEL_ID_DOMAIN);
        hash_field(&mut hasher, &self.assembly_id.0);
        hash_field(&mut hasher, &self.model_family.canonical_tag());
        hash_field(&mut hasher, &self.applicability_profile_id);
        for mut values in [
            self.source_evidence_refs.clone(),
            self.geometry_binding_refs.clone(),
            self.material_evidence_refs.clone(),
        ] {
            values.sort();
            for value in values {
                hash_field(&mut hasher, &value);
            }
            hasher.update(&[0xff]);
        }
        Ok(MagneticModelProfileId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MagneticEvidencePlaneV1 {
    DeclaredRated,
    ConfiguredCommanded,
    Predicted,
    InferredDerived,
    ObservedMeasured,
}

impl MagneticEvidencePlaneV1 {
    pub fn satisfies_exact(self, required: Self) -> Result<(), MagneticError> {
        if self == required {
            Ok(())
        } else {
            Err(MagneticError::EvidencePlaneMismatch)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagneticEvidenceRefV1 {
    pub evidence_id: String,
    pub assembly_id: MagneticAssemblyId,
    pub plane: MagneticEvidencePlaneV1,
    pub semantic_kind_id: String,
    pub member_id: Option<String>,
    pub model_profile_id: Option<MagneticModelProfileId>,
}

impl MagneticEvidenceRefV1 {
    pub fn validate_for(&self, assembly: &MagneticAssemblyV1) -> Result<(), MagneticError> {
        validate_token("evidence_id", &self.evidence_id)?;
        validate_token("semantic_kind_id", &self.semantic_kind_id)?;
        if self.assembly_id != assembly.assembly_id()? {
            return Err(MagneticError::EvidenceAssemblyMismatch);
        }
        if let Some(member_id) = self.member_id.as_deref() {
            validate_token("member_id", member_id)?;
            if !assembly.members.iter().any(|m| m.member_id == member_id) {
                return Err(MagneticError::UnknownMember(member_id.into()));
            }
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        format!(
            "{}|{}|{:?}|{}|{}|{}",
            self.evidence_id,
            self.assembly_id.0,
            self.plane,
            self.semantic_kind_id,
            self.member_id.as_deref().unwrap_or("none"),
            self.model_profile_id
                .as_ref()
                .map(|id| id.0.as_str())
                .unwrap_or("none")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagneticEvidenceSnapshotV1 {
    pub assembly: MagneticAssemblyV1,
    #[serde(default)]
    pub model_profiles: Vec<MagneticModelProfileV1>,
    #[serde(default)]
    pub evidence_refs: Vec<MagneticEvidenceRefV1>,
}

impl MagneticEvidenceSnapshotV1 {
    pub fn snapshot_id(&self) -> Result<MagneticEvidenceSnapshotId, MagneticError> {
        let assembly_id = self.assembly.assembly_id()?;
        let mut models = BTreeMap::new();
        for profile in &self.model_profiles {
            let id = profile.model_profile_id(&self.assembly)?;
            if models.insert(id.0.clone(), profile).is_some() {
                return Err(MagneticError::DuplicateModelProfile(id.0));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        let mut evidence_keys = Vec::new();
        for evidence in &self.evidence_refs {
            evidence.validate_for(&self.assembly)?;
            if !evidence_ids.insert(evidence.evidence_id.clone()) {
                return Err(MagneticError::DuplicateEvidence(evidence.evidence_id.clone()));
            }

            let model = match &evidence.model_profile_id {
                Some(id) => Some(
                    models
                        .get(&id.0)
                        .copied()
                        .ok_or_else(|| MagneticError::UnknownModelProfile(id.0.clone()))?,
                ),
                None => None,
            };

            match evidence.plane {
                MagneticEvidencePlaneV1::Predicted => {
                    let model = model.ok_or(MagneticError::PredictedEvidenceRequiresModelProfile)?;
                    if model.model_family.is_backend_requirement() {
                        return Err(MagneticError::RequirementMarkerCannotBackPrediction);
                    }
                    if model.model_family == MagneticModelFamilyV1::ImportedPredictedFieldMap {
                        return Err(MagneticError::ImportedPredictionNotClosed);
                    }
                    if !model.model_family.supports_prediction() {
                        return Err(MagneticError::RequirementMarkerCannotBackPrediction);
                    }
                }
                MagneticEvidencePlaneV1::ObservedMeasured if model.is_some() => {
                    return Err(MagneticError::MeasuredEvidenceCannotUseModelProfile);
                }
                _ => {}
            }
            evidence_keys.push(evidence.canonical_key());
        }
        evidence_keys.sort();

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, SNAPSHOT_ID_DOMAIN);
        hash_field(&mut hasher, &assembly_id.0);
        for id in models.keys() {
            hash_field(&mut hasher, id);
        }
        for evidence in evidence_keys {
            hash_field(&mut hasher, &evidence);
        }
        Ok(MagneticEvidenceSnapshotId(
            hasher.finalize().to_hex().to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid(c: char) -> ComponentSubjectId {
        ComponentSubjectId(std::iter::repeat_n(c, 64).collect())
    }

    fn assembly() -> MagneticAssemblyV1 {
        MagneticAssemblyV1 {
            assembly_kind_id: "magnetic-reference".into(),
            members: vec![
                MagneticMemberV1 {
                    member_id: "magnet".into(),
                    role: MagneticMemberRoleV1::PermanentMagnet,
                    component_subject_id: Some(cid('a')),
                    geometry_binding_id: Some("geom:magnet".into()),
                    material_state_ref: Some("mat:magnet".into()),
                    orientation_profile_id: Some("orientation:axial".into()),
                    display_name: Some("Magnet".into()),
                },
                MagneticMemberV1 {
                    member_id: "field".into(),
                    role: MagneticMemberRoleV1::FieldRegion,
                    component_subject_id: None,
                    geometry_binding_id: Some("region:field".into()),
                    material_state_ref: None,
                    orientation_profile_id: None,
                    display_name: Some("Field".into()),
                },
            ],
            relations: vec![MagneticRelationV1 {
                relation_id: "bias".into(),
                kind: MagneticRelationKindV1::PermanentMagnetBiasesRegion,
                from_member_id: "magnet".into(),
                to_member_id: "field".into(),
                coupling_profile_id: Some("bias:v1".into()),
            }],
            display_name: Some("Reference".into()),
        }
    }

    fn profile(assembly: &MagneticAssemblyV1, family: MagneticModelFamilyV1) -> MagneticModelProfileV1 {
        MagneticModelProfileV1 {
            assembly_id: assembly.assembly_id().unwrap(),
            model_family: family,
            applicability_profile_id: "fixture:v1".into(),
            source_evidence_refs: vec!["source:model".into()],
            geometry_binding_refs: vec![],
            material_evidence_refs: vec![],
        }
    }

    #[test]
    fn label_change_is_nonsemantic_but_orientation_change_is_semantic() {
        let a = assembly();
        let mut b = a.clone();
        b.display_name = Some("Renamed".into());
        assert_eq!(a.assembly_id().unwrap(), b.assembly_id().unwrap());
        b.members[0].orientation_profile_id = Some("orientation:reversed".into());
        assert_ne!(a.assembly_id().unwrap(), b.assembly_id().unwrap());
    }

    #[test]
    fn external_fem_requirement_cannot_mint_prediction() {
        let assembly = assembly();
        let model = profile(&assembly, MagneticModelFamilyV1::ExternalFiniteElementRequired);
        let model_id = model.model_profile_id(&assembly).unwrap();
        let snapshot = MagneticEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![MagneticEvidenceRefV1 {
                evidence_id: "prediction:false".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: MagneticEvidencePlaneV1::Predicted,
                semantic_kind_id: "magnetic.field".into(),
                member_id: Some("field".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(MagneticError::RequirementMarkerCannotBackPrediction)
        );
    }

    #[test]
    fn dangling_and_duplicate_model_profiles_fail_closed() {
        let assembly = assembly();
        let dangling = MagneticEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![],
            evidence_refs: vec![MagneticEvidenceRefV1 {
                evidence_id: "prediction:dangling".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: MagneticEvidencePlaneV1::Predicted,
                semantic_kind_id: "magnetic.field".into(),
                member_id: Some("field".into()),
                model_profile_id: Some(MagneticModelProfileId("missing".into())),
            }],
        };
        assert!(matches!(
            dangling.snapshot_id(),
            Err(MagneticError::UnknownModelProfile(_))
        ));

        let model = profile(&assembly, MagneticModelFamilyV1::LinearReluctance);
        let duplicate = MagneticEvidenceSnapshotV1 {
            assembly,
            model_profiles: vec![model.clone(), model],
            evidence_refs: vec![],
        };
        assert!(matches!(
            duplicate.snapshot_id(),
            Err(MagneticError::DuplicateModelProfile(_))
        ));
    }

    #[test]
    fn imported_predicted_field_requires_result_closure() {
        let assembly = assembly();
        let model = profile(&assembly, MagneticModelFamilyV1::ImportedPredictedFieldMap);
        let model_id = model.model_profile_id(&assembly).unwrap();
        let snapshot = MagneticEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![MagneticEvidenceRefV1 {
                evidence_id: "prediction:imported".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: MagneticEvidencePlaneV1::Predicted,
                semantic_kind_id: "magnetic.field-map".into(),
                member_id: Some("field".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(MagneticError::ImportedPredictionNotClosed)
        );
    }

    #[test]
    fn measured_field_cannot_use_model_profile_as_measurement_authority() {
        let assembly = assembly();
        let model = profile(&assembly, MagneticModelFamilyV1::ImportedMeasuredFieldMap);
        let model_id = model.model_profile_id(&assembly).unwrap();
        let snapshot = MagneticEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![MagneticEvidenceRefV1 {
                evidence_id: "measurement:field".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: MagneticEvidencePlaneV1::ObservedMeasured,
                semantic_kind_id: "magnetic.field-observation".into(),
                member_id: Some("field".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(MagneticError::MeasuredEvidenceCannotUseModelProfile)
        );
    }

    #[test]
    fn valid_analytical_prediction_is_still_only_prediction() {
        let assembly = assembly();
        let model = profile(&assembly, MagneticModelFamilyV1::LinearReluctance);
        let model_id = model.model_profile_id(&assembly).unwrap();
        let snapshot = MagneticEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![MagneticEvidenceRefV1 {
                evidence_id: "prediction:field".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: MagneticEvidencePlaneV1::Predicted,
                semantic_kind_id: "magnetic.field".into(),
                member_id: Some("field".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert!(snapshot.snapshot_id().is_ok());
        assert_ne!(
            MagneticEvidencePlaneV1::Predicted,
            MagneticEvidencePlaneV1::ObservedMeasured
        );
    }

    #[test]
    fn serialization_does_not_bypass_dangling_reference_validation() {
        let assembly = assembly();
        let snapshot = MagneticEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![],
            evidence_refs: vec![MagneticEvidenceRefV1 {
                evidence_id: "prediction:dangling".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: MagneticEvidencePlaneV1::Predicted,
                semantic_kind_id: "magnetic.field".into(),
                member_id: None,
                model_profile_id: Some(MagneticModelProfileId("missing".into())),
            }],
        };
        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: MagneticEvidenceSnapshotV1 = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.snapshot_id().is_err());
    }
}
