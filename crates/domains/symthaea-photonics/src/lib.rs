// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance-bound photonic assembly and model semantics.
//!
//! This crate owns neither canonical quantities nor optical solvers nor physical optical
//! observations/actuation. It defines exact photonic assembly/model subjects that may later be
//! consumed by analytical optics, numerical backends, FIELD, and HAL.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_engineering_catalog::ComponentSubjectId;
use thiserror::Error;

const ASSEMBLY_DOMAIN: &str = "symthaea-photonics::assembly-v1";
const MODEL_DOMAIN: &str = "symthaea-photonics::model-profile-v1";
const SNAPSHOT_DOMAIN: &str = "symthaea-photonics::evidence-snapshot-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PhotonicsError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("{field} contains leading or trailing whitespace")]
    NonCanonicalWhitespace { field: &'static str },
    #[error("duplicate photonic member id: {0}")]
    DuplicateMember(String),
    #[error("duplicate photonic relation id: {0}")]
    DuplicateRelation(String),
    #[error("photonic relation references unknown member: {0}")]
    UnknownMember(String),
    #[error("photonic relation cannot self-reference: {0}")]
    SelfRelation(String),
    #[error("physical photonic role requires a catalog component subject: {0}")]
    ComponentRequired(String),
    #[error("field region must not masquerade as a catalog component: {0}")]
    FieldRegionCannotBeComponent(String),
    #[error("model profile is bound to a different photonic assembly")]
    AssemblyMismatch,
    #[error("model profile requires source/model evidence")]
    PredictiveEvidenceRequired,
    #[error("unknown/not-admitted model cannot claim model authority")]
    UnknownModelNotPredictive,
    #[error("selected model family cannot support active-source prediction")]
    ActiveSourceCapabilityMismatch,
    #[error("active-source prediction requires an explicit gain medium")]
    GainMediumRequired,
    #[error("active-source prediction requires an explicit source/emitter candidate")]
    SourceCandidateRequired,
    #[error("evidence reference is bound to a different assembly")]
    EvidenceAssemblyMismatch,
    #[error("evidence references an unknown member: {0}")]
    EvidenceUnknownMember(String),
    #[error("evidence plane mismatch")]
    EvidencePlaneMismatch,
    #[error("duplicate photonic model-profile identity: {0}")]
    DuplicateModelProfile(String),
    #[error("evidence references a model profile absent from the snapshot: {0}")]
    UnknownModelProfile(String),
    #[error("duplicate evidence identity: {0}")]
    DuplicateEvidence(String),
    #[error("predicted evidence requires an exact local model-profile reference")]
    PredictedEvidenceRequiresModelProfile,
    #[error("model family is a capability/result requirement and cannot itself back prediction evidence")]
    RequirementMarkerCannotBackPrediction,
    #[error("imported numerical field/mode requires result-receipt closure before it can back admitted prediction evidence")]
    ImportedNumericalResultNotClosed,
    #[error("observed/measured evidence must not use a prediction model profile as its measurement authority")]
    MeasuredEvidenceCannotUseModelProfile,
}

fn validate_token(field: &'static str, value: &str) -> Result<(), PhotonicsError> {
    if value.is_empty() {
        return Err(PhotonicsError::EmptyField { field });
    }
    if value.trim() != value {
        return Err(PhotonicsError::NonCanonicalWhitespace { field });
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
pub struct PhotonicAssemblyId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PhotonicModelProfileId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PhotonicEvidenceSnapshotId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhotonicMemberRoleV1 {
    SourceCandidate,
    GainMedium,
    PumpSource,
    ResonatorMirror,
    PassiveOptic,
    Lens,
    Waveguide,
    Fiber,
    CouplerSplitterCombiner,
    Modulator,
    IsolatorOrNonreciprocal,
    Filter,
    PolarizerOrAnalyzer,
    ApertureOrStop,
    Detector,
    OpticalInterface,
    ThermalReference,
    MechanicalAlignmentReference,
    FieldRegion,
    Other(String),
}

impl PhotonicMemberRoleV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::SourceCandidate => "source-candidate".into(),
            Self::GainMedium => "gain-medium".into(),
            Self::PumpSource => "pump-source".into(),
            Self::ResonatorMirror => "resonator-mirror".into(),
            Self::PassiveOptic => "passive-optic".into(),
            Self::Lens => "lens".into(),
            Self::Waveguide => "waveguide".into(),
            Self::Fiber => "fiber".into(),
            Self::CouplerSplitterCombiner => "coupler-splitter-combiner".into(),
            Self::Modulator => "modulator".into(),
            Self::IsolatorOrNonreciprocal => "isolator-or-nonreciprocal".into(),
            Self::Filter => "filter".into(),
            Self::PolarizerOrAnalyzer => "polarizer-or-analyzer".into(),
            Self::ApertureOrStop => "aperture-or-stop".into(),
            Self::Detector => "detector".into(),
            Self::OpticalInterface => "optical-interface".into(),
            Self::ThermalReference => "thermal-reference".into(),
            Self::MechanicalAlignmentReference => "mechanical-alignment-reference".into(),
            Self::FieldRegion => "field-region".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), PhotonicsError> {
        if let Self::Other(value) = self {
            validate_token("member_role.other", value)?;
        }
        Ok(())
    }

    fn component_required(&self) -> bool {
        !matches!(
            self,
            Self::FieldRegion
                | Self::OpticalInterface
                | Self::ThermalReference
                | Self::MechanicalAlignmentReference
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotonicMemberV1 {
    pub member_id: String,
    pub role: PhotonicMemberRoleV1,
    pub component_subject_id: Option<ComponentSubjectId>,
    pub geometry_binding_id: Option<String>,
    pub material_state_ref: Option<String>,
    pub alignment_profile_id: Option<String>,
    pub external_coupling_ref: Option<String>,
    pub display_name: Option<String>,
}

impl PhotonicMemberV1 {
    pub fn validate(&self) -> Result<(), PhotonicsError> {
        validate_token("member_id", &self.member_id)?;
        self.role.validate()?;
        if self.role.component_required() && self.component_subject_id.is_none() {
            return Err(PhotonicsError::ComponentRequired(self.member_id.clone()));
        }
        if self.role == PhotonicMemberRoleV1::FieldRegion && self.component_subject_id.is_some() {
            return Err(PhotonicsError::FieldRegionCannotBeComponent(self.member_id.clone()));
        }
        for (field, value) in [
            ("geometry_binding_id", self.geometry_binding_id.as_deref()),
            ("material_state_ref", self.material_state_ref.as_deref()),
            ("alignment_profile_id", self.alignment_profile_id.as_deref()),
            ("external_coupling_ref", self.external_coupling_ref.as_deref()),
        ] {
            if let Some(value) = value {
                validate_token(field, value)?;
            }
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            self.member_id,
            self.role.canonical_tag(),
            self.component_subject_id
                .as_ref()
                .map(|id| id.0.as_str())
                .unwrap_or("none"),
            self.geometry_binding_id.as_deref().unwrap_or("none"),
            self.material_state_ref.as_deref().unwrap_or("none"),
            self.alignment_profile_id.as_deref().unwrap_or("none"),
            self.external_coupling_ref.as_deref().unwrap_or("none")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhotonicRelationKindV1 {
    OpticalPath,
    ResonatorFeedbackPath,
    PumpCoupling,
    GainInteraction,
    DetectorObservationPath,
    ModulationRelation,
    PolarizationTransform,
    WaveguideOrFiberCoupling,
    ThermalCouplingRef,
    MechanicalAlignmentRef,
    MagnetoOpticCouplingRef,
    Other(String),
}

impl PhotonicRelationKindV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::OpticalPath => "optical-path".into(),
            Self::ResonatorFeedbackPath => "resonator-feedback-path".into(),
            Self::PumpCoupling => "pump-coupling".into(),
            Self::GainInteraction => "gain-interaction".into(),
            Self::DetectorObservationPath => "detector-observation-path".into(),
            Self::ModulationRelation => "modulation-relation".into(),
            Self::PolarizationTransform => "polarization-transform".into(),
            Self::WaveguideOrFiberCoupling => "waveguide-or-fiber-coupling".into(),
            Self::ThermalCouplingRef => "thermal-coupling-ref".into(),
            Self::MechanicalAlignmentRef => "mechanical-alignment-ref".into(),
            Self::MagnetoOpticCouplingRef => "magneto-optic-coupling-ref".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), PhotonicsError> {
        if let Self::Other(value) = self {
            validate_token("relation_kind.other", value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotonicRelationV1 {
    pub relation_id: String,
    pub kind: PhotonicRelationKindV1,
    pub from_member_id: String,
    pub to_member_id: String,
    pub coupling_profile_id: Option<String>,
}

impl PhotonicRelationV1 {
    fn validate(&self) -> Result<(), PhotonicsError> {
        validate_token("relation_id", &self.relation_id)?;
        validate_token("from_member_id", &self.from_member_id)?;
        validate_token("to_member_id", &self.to_member_id)?;
        self.kind.validate()?;
        if let Some(value) = self.coupling_profile_id.as_deref() {
            validate_token("coupling_profile_id", value)?;
        }
        if self.from_member_id == self.to_member_id {
            return Err(PhotonicsError::SelfRelation(self.relation_id.clone()));
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
pub struct PhotonicAssemblyV1 {
    pub assembly_kind_id: String,
    #[serde(default)]
    pub members: Vec<PhotonicMemberV1>,
    #[serde(default)]
    pub relations: Vec<PhotonicRelationV1>,
    pub display_name: Option<String>,
}

impl PhotonicAssemblyV1 {
    pub fn validate(&self) -> Result<(), PhotonicsError> {
        validate_token("assembly_kind_id", &self.assembly_kind_id)?;
        let mut members = BTreeSet::new();
        for member in &self.members {
            member.validate()?;
            if !members.insert(member.member_id.clone()) {
                return Err(PhotonicsError::DuplicateMember(member.member_id.clone()));
            }
        }
        let mut relations = BTreeSet::new();
        for relation in &self.relations {
            relation.validate()?;
            if !relations.insert(relation.relation_id.clone()) {
                return Err(PhotonicsError::DuplicateRelation(relation.relation_id.clone()));
            }
            if !members.contains(&relation.from_member_id) {
                return Err(PhotonicsError::UnknownMember(relation.from_member_id.clone()));
            }
            if !members.contains(&relation.to_member_id) {
                return Err(PhotonicsError::UnknownMember(relation.to_member_id.clone()));
            }
        }
        Ok(())
    }

    pub fn has_role(&self, role: PhotonicMemberRoleV1) -> bool {
        self.members.iter().any(|member| member.role == role)
    }

    pub fn assembly_id(&self) -> Result<PhotonicAssemblyId, PhotonicsError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, ASSEMBLY_DOMAIN);
        hash_field(&mut hasher, &self.assembly_kind_id);
        let mut members = self
            .members
            .iter()
            .map(PhotonicMemberV1::canonical_key)
            .collect::<Vec<_>>();
        members.sort();
        for member in members {
            hash_field(&mut hasher, &member);
        }
        let mut relations = self
            .relations
            .iter()
            .map(PhotonicRelationV1::canonical_key)
            .collect::<Vec<_>>();
        relations.sort();
        for relation in relations {
            hash_field(&mut hasher, &relation);
        }
        Ok(PhotonicAssemblyId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhotonicModelFamilyV1 {
    GeometricRay,
    GaussianParaxial,
    AbcdResonator,
    ReducedOrderActiveGain,
    WaveguideEigenmode,
    FullWaveFdtdRequired,
    PhotonicBandEigenmodeRequired,
    ThermoOpticOptomechanicalCoupled,
    ImportedNumericalFieldOrMode,
    UnknownNotAdmitted,
    Other(String),
}

impl PhotonicModelFamilyV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::GeometricRay => "geometric-ray".into(),
            Self::GaussianParaxial => "gaussian-paraxial".into(),
            Self::AbcdResonator => "abcd-resonator".into(),
            Self::ReducedOrderActiveGain => "reduced-order-active-gain".into(),
            Self::WaveguideEigenmode => "waveguide-eigenmode".into(),
            Self::FullWaveFdtdRequired => "full-wave-fdtd-required".into(),
            Self::PhotonicBandEigenmodeRequired => "photonic-band-eigenmode-required".into(),
            Self::ThermoOpticOptomechanicalCoupled => {
                "thermo-optic-optomechanical-coupled".into()
            }
            Self::ImportedNumericalFieldOrMode => "imported-numerical-field-or-mode".into(),
            Self::UnknownNotAdmitted => "unknown-not-admitted".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), PhotonicsError> {
        if let Self::Other(value) = self {
            validate_token("model_family.other", value)?;
        }
        Ok(())
    }

    pub fn supports_active_source_prediction(&self) -> bool {
        matches!(self, Self::ReducedOrderActiveGain)
    }

    pub fn is_backend_requirement(&self) -> bool {
        matches!(
            self,
            Self::FullWaveFdtdRequired | Self::PhotonicBandEigenmodeRequired
        )
    }

    pub fn supports_prediction(&self) -> bool {
        !matches!(
            self,
            Self::UnknownNotAdmitted
                | Self::FullWaveFdtdRequired
                | Self::PhotonicBandEigenmodeRequired
                | Self::ImportedNumericalFieldOrMode
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhotonicPredictionCapabilityV1 {
    PassiveTransferOnly,
    ActiveSourceCandidate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotonicModelProfileV1 {
    pub assembly_id: PhotonicAssemblyId,
    pub model_family: PhotonicModelFamilyV1,
    pub prediction_capability: PhotonicPredictionCapabilityV1,
    pub applicability_profile_id: String,
    #[serde(default)]
    pub source_evidence_refs: Vec<String>,
    #[serde(default)]
    pub geometry_binding_refs: Vec<String>,
    #[serde(default)]
    pub material_evidence_refs: Vec<String>,
}

impl PhotonicModelProfileV1 {
    pub fn validate_for(&self, assembly: &PhotonicAssemblyV1) -> Result<(), PhotonicsError> {
        self.model_family.validate()?;
        validate_token("applicability_profile_id", &self.applicability_profile_id)?;
        if self.assembly_id != assembly.assembly_id()? {
            return Err(PhotonicsError::AssemblyMismatch);
        }
        if self.model_family == PhotonicModelFamilyV1::UnknownNotAdmitted {
            return Err(PhotonicsError::UnknownModelNotPredictive);
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
            return Err(PhotonicsError::PredictiveEvidenceRequired);
        }
        if self.prediction_capability == PhotonicPredictionCapabilityV1::ActiveSourceCandidate {
            if !self.model_family.supports_active_source_prediction() {
                return Err(PhotonicsError::ActiveSourceCapabilityMismatch);
            }
            if !assembly.has_role(PhotonicMemberRoleV1::GainMedium) {
                return Err(PhotonicsError::GainMediumRequired);
            }
            if !assembly.has_role(PhotonicMemberRoleV1::SourceCandidate) {
                return Err(PhotonicsError::SourceCandidateRequired);
            }
        }
        Ok(())
    }

    pub fn model_profile_id(
        &self,
        assembly: &PhotonicAssemblyV1,
    ) -> Result<PhotonicModelProfileId, PhotonicsError> {
        self.validate_for(assembly)?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, MODEL_DOMAIN);
        hash_field(&mut hasher, &self.assembly_id.0);
        hash_field(&mut hasher, &self.model_family.canonical_tag());
        hash_field(&mut hasher, &format!("{:?}", self.prediction_capability));
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
        Ok(PhotonicModelProfileId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhotonicEvidencePlaneV1 {
    DeclaredRated,
    ConfiguredCommanded,
    Predicted,
    InferredDerived,
    ObservedMeasured,
}

impl PhotonicEvidencePlaneV1 {
    pub fn satisfies_exact(self, required: Self) -> Result<(), PhotonicsError> {
        if self == required {
            Ok(())
        } else {
            Err(PhotonicsError::EvidencePlaneMismatch)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotonicEvidenceRefV1 {
    pub evidence_id: String,
    pub assembly_id: PhotonicAssemblyId,
    pub plane: PhotonicEvidencePlaneV1,
    pub semantic_kind_id: String,
    pub member_id: Option<String>,
    pub model_profile_id: Option<PhotonicModelProfileId>,
}

impl PhotonicEvidenceRefV1 {
    pub fn validate_for(&self, assembly: &PhotonicAssemblyV1) -> Result<(), PhotonicsError> {
        validate_token("evidence_id", &self.evidence_id)?;
        validate_token("semantic_kind_id", &self.semantic_kind_id)?;
        if self.assembly_id != assembly.assembly_id()? {
            return Err(PhotonicsError::EvidenceAssemblyMismatch);
        }
        if let Some(member_id) = self.member_id.as_deref() {
            validate_token("member_id", member_id)?;
            if !assembly.members.iter().any(|member| member.member_id == member_id) {
                return Err(PhotonicsError::EvidenceUnknownMember(member_id.into()));
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
pub struct PhotonicEvidenceSnapshotV1 {
    pub assembly: PhotonicAssemblyV1,
    #[serde(default)]
    pub model_profiles: Vec<PhotonicModelProfileV1>,
    #[serde(default)]
    pub evidence_refs: Vec<PhotonicEvidenceRefV1>,
}

impl PhotonicEvidenceSnapshotV1 {
    pub fn snapshot_id(&self) -> Result<PhotonicEvidenceSnapshotId, PhotonicsError> {
        let assembly_id = self.assembly.assembly_id()?;
        let mut models = BTreeMap::new();
        for model in &self.model_profiles {
            let id = model.model_profile_id(&self.assembly)?;
            if models.insert(id.0.clone(), model).is_some() {
                return Err(PhotonicsError::DuplicateModelProfile(id.0));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        let mut evidence = Vec::new();
        for item in &self.evidence_refs {
            item.validate_for(&self.assembly)?;
            if !evidence_ids.insert(item.evidence_id.clone()) {
                return Err(PhotonicsError::DuplicateEvidence(item.evidence_id.clone()));
            }

            let model = match &item.model_profile_id {
                Some(id) => Some(
                    models
                        .get(&id.0)
                        .copied()
                        .ok_or_else(|| PhotonicsError::UnknownModelProfile(id.0.clone()))?,
                ),
                None => None,
            };

            match item.plane {
                PhotonicEvidencePlaneV1::Predicted => {
                    let model = model.ok_or(PhotonicsError::PredictedEvidenceRequiresModelProfile)?;
                    if model.model_family.is_backend_requirement() {
                        return Err(PhotonicsError::RequirementMarkerCannotBackPrediction);
                    }
                    if model.model_family == PhotonicModelFamilyV1::ImportedNumericalFieldOrMode {
                        return Err(PhotonicsError::ImportedNumericalResultNotClosed);
                    }
                    if !model.model_family.supports_prediction() {
                        return Err(PhotonicsError::RequirementMarkerCannotBackPrediction);
                    }
                }
                PhotonicEvidencePlaneV1::ObservedMeasured if model.is_some() => {
                    return Err(PhotonicsError::MeasuredEvidenceCannotUseModelProfile);
                }
                _ => {}
            }

            evidence.push(item.canonical_key());
        }
        evidence.sort();

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, SNAPSHOT_DOMAIN);
        hash_field(&mut hasher, &assembly_id.0);
        for id in models.keys() {
            hash_field(&mut hasher, id);
        }
        for item in evidence {
            hash_field(&mut hasher, &item);
        }
        Ok(PhotonicEvidenceSnapshotId(
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

    fn passive_cavity() -> PhotonicAssemblyV1 {
        PhotonicAssemblyV1 {
            assembly_kind_id: "passive-reference-cavity".into(),
            members: vec![
                PhotonicMemberV1 {
                    member_id: "mirror-a".into(),
                    role: PhotonicMemberRoleV1::ResonatorMirror,
                    component_subject_id: Some(cid('a')),
                    geometry_binding_id: Some("geom:mirror-a".into()),
                    material_state_ref: Some("surface:coating-a".into()),
                    alignment_profile_id: Some("align:a".into()),
                    external_coupling_ref: None,
                    display_name: Some("Input mirror".into()),
                },
                PhotonicMemberV1 {
                    member_id: "mirror-b".into(),
                    role: PhotonicMemberRoleV1::ResonatorMirror,
                    component_subject_id: Some(cid('b')),
                    geometry_binding_id: Some("geom:mirror-b".into()),
                    material_state_ref: Some("surface:coating-b".into()),
                    alignment_profile_id: Some("align:b".into()),
                    external_coupling_ref: None,
                    display_name: Some("End mirror".into()),
                },
                PhotonicMemberV1 {
                    member_id: "cavity-field".into(),
                    role: PhotonicMemberRoleV1::FieldRegion,
                    component_subject_id: None,
                    geometry_binding_id: Some("region:cavity".into()),
                    material_state_ref: None,
                    alignment_profile_id: None,
                    external_coupling_ref: None,
                    display_name: Some("Cavity field".into()),
                },
            ],
            relations: vec![
                PhotonicRelationV1 {
                    relation_id: "feedback-a".into(),
                    kind: PhotonicRelationKindV1::ResonatorFeedbackPath,
                    from_member_id: "mirror-a".into(),
                    to_member_id: "cavity-field".into(),
                    coupling_profile_id: Some("reflection:a".into()),
                },
                PhotonicRelationV1 {
                    relation_id: "feedback-b".into(),
                    kind: PhotonicRelationKindV1::ResonatorFeedbackPath,
                    from_member_id: "cavity-field".into(),
                    to_member_id: "mirror-b".into(),
                    coupling_profile_id: Some("reflection:b".into()),
                },
            ],
            display_name: Some("Reference cavity".into()),
        }
    }

    fn model(assembly: &PhotonicAssemblyV1, family: PhotonicModelFamilyV1) -> PhotonicModelProfileV1 {
        PhotonicModelProfileV1 {
            assembly_id: assembly.assembly_id().unwrap(),
            model_family: family,
            prediction_capability: PhotonicPredictionCapabilityV1::PassiveTransferOnly,
            applicability_profile_id: "fixture:v1".into(),
            source_evidence_refs: vec!["model:fixture".into()],
            geometry_binding_refs: vec![],
            material_evidence_refs: vec![],
        }
    }

    #[test]
    fn labels_do_not_change_identity_but_topology_does() {
        let a = passive_cavity();
        let mut b = a.clone();
        b.display_name = Some("renamed".into());
        assert_eq!(a.assembly_id().unwrap(), b.assembly_id().unwrap());
        b.relations[0].coupling_profile_id = Some("changed".into());
        assert_ne!(a.assembly_id().unwrap(), b.assembly_id().unwrap());
    }

    #[test]
    fn backend_requirement_is_not_prediction_authority() {
        let assembly = passive_cavity();
        let requirement = model(&assembly, PhotonicModelFamilyV1::FullWaveFdtdRequired);
        let requirement_id = requirement.model_profile_id(&assembly).unwrap();
        let snapshot = PhotonicEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![requirement],
            evidence_refs: vec![PhotonicEvidenceRefV1 {
                evidence_id: "prediction:false".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: PhotonicEvidencePlaneV1::Predicted,
                semantic_kind_id: "optical.field".into(),
                member_id: Some("cavity-field".into()),
                model_profile_id: Some(requirement_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(PhotonicsError::RequirementMarkerCannotBackPrediction)
        );
    }

    #[test]
    fn dangling_model_reference_rejects() {
        let assembly = passive_cavity();
        let snapshot = PhotonicEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![],
            evidence_refs: vec![PhotonicEvidenceRefV1 {
                evidence_id: "prediction:dangling".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: PhotonicEvidencePlaneV1::Predicted,
                semantic_kind_id: "optical.mode".into(),
                member_id: Some("cavity-field".into()),
                model_profile_id: Some(PhotonicModelProfileId("missing".into())),
            }],
        };
        assert!(matches!(
            snapshot.snapshot_id(),
            Err(PhotonicsError::UnknownModelProfile(_))
        ));
    }

    #[test]
    fn duplicate_model_profile_rejects_instead_of_collapsing() {
        let assembly = passive_cavity();
        let profile = model(&assembly, PhotonicModelFamilyV1::AbcdResonator);
        let snapshot = PhotonicEvidenceSnapshotV1 {
            assembly,
            model_profiles: vec![profile.clone(), profile],
            evidence_refs: vec![],
        };
        assert!(matches!(
            snapshot.snapshot_id(),
            Err(PhotonicsError::DuplicateModelProfile(_))
        ));
    }

    #[test]
    fn measured_evidence_cannot_use_prediction_model_as_authority() {
        let assembly = passive_cavity();
        let profile = model(&assembly, PhotonicModelFamilyV1::AbcdResonator);
        let profile_id = profile.model_profile_id(&assembly).unwrap();
        let snapshot = PhotonicEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![profile],
            evidence_refs: vec![PhotonicEvidenceRefV1 {
                evidence_id: "measurement:cavity".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: PhotonicEvidencePlaneV1::ObservedMeasured,
                semantic_kind_id: "optical.detector-observation".into(),
                member_id: Some("cavity-field".into()),
                model_profile_id: Some(profile_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(PhotonicsError::MeasuredEvidenceCannotUseModelProfile)
        );
    }

    #[test]
    fn valid_analytical_prediction_remains_admitted_as_prediction_only() {
        let assembly = passive_cavity();
        let profile = model(&assembly, PhotonicModelFamilyV1::AbcdResonator);
        let profile_id = profile.model_profile_id(&assembly).unwrap();
        let snapshot = PhotonicEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![profile],
            evidence_refs: vec![PhotonicEvidenceRefV1 {
                evidence_id: "prediction:cavity".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: PhotonicEvidencePlaneV1::Predicted,
                semantic_kind_id: "optical.cavity-mode".into(),
                member_id: Some("cavity-field".into()),
                model_profile_id: Some(profile_id),
            }],
        };
        assert!(snapshot.snapshot_id().is_ok());
    }

    #[test]
    fn serialization_does_not_bypass_dangling_reference_validation() {
        let assembly = passive_cavity();
        let snapshot = PhotonicEvidenceSnapshotV1 {
            assembly: assembly.clone(),
            model_profiles: vec![],
            evidence_refs: vec![PhotonicEvidenceRefV1 {
                evidence_id: "prediction:dangling".into(),
                assembly_id: assembly.assembly_id().unwrap(),
                plane: PhotonicEvidencePlaneV1::Predicted,
                semantic_kind_id: "optical.mode".into(),
                member_id: None,
                model_profile_id: Some(PhotonicModelProfileId("missing".into())),
            }],
        };
        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: PhotonicEvidenceSnapshotV1 = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.snapshot_id().is_err());
    }
}
