// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reusable thermal-network, boundary and evidence semantics.
//!
//! This crate does not own canonical physical quantities, material-property truth, CFD/FEA
//! solvers, FIELD observations, or physical heater/fan/pump authority. It defines exact thermal
//! topology/model subjects that those systems can reference.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_engineering_catalog::ComponentSubjectId;
use thiserror::Error;

const NETWORK_DOMAIN: &str = "symthaea-thermal-engineering::network-v1";
const MODEL_DOMAIN: &str = "symthaea-thermal-engineering::model-profile-v1";
const SNAPSHOT_DOMAIN: &str = "symthaea-thermal-engineering::evidence-snapshot-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ThermalError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("{field} contains leading or trailing whitespace")]
    NonCanonicalWhitespace { field: &'static str },
    #[error("duplicate thermal member id: {0}")]
    DuplicateMember(String),
    #[error("duplicate thermal relation id: {0}")]
    DuplicateRelation(String),
    #[error("thermal relation references unknown member: {0}")]
    UnknownMember(String),
    #[error("thermal relation cannot self-reference: {0}")]
    SelfRelation(String),
    #[error("physical thermal role requires a catalog component subject: {0}")]
    ComponentRequired(String),
    #[error("abstract boundary/observation region must not masquerade as a catalog component: {0}")]
    AbstractRegionCannotBeComponent(String),
    #[error("thermal model profile is bound to a different network")]
    NetworkMismatch,
    #[error("thermal model profile requires exact source/model evidence")]
    PredictiveEvidenceRequired,
    #[error("unknown/not-admitted thermal model cannot claim model authority")]
    UnknownModelNotPredictive,
    #[error("evidence reference is bound to a different network")]
    EvidenceNetworkMismatch,
    #[error("evidence references an unknown member: {0}")]
    EvidenceUnknownMember(String),
    #[error("evidence plane mismatch")]
    EvidencePlaneMismatch,
    #[error("duplicate thermal model-profile identity: {0}")]
    DuplicateModelProfile(String),
    #[error("evidence references a model profile absent from the snapshot: {0}")]
    UnknownModelProfile(String),
    #[error("duplicate thermal evidence identity: {0}")]
    DuplicateEvidence(String),
    #[error("predicted thermal evidence requires an exact local model-profile reference")]
    PredictedEvidenceRequiresModelProfile,
    #[error("external solver requirement cannot itself back thermal prediction evidence")]
    RequirementMarkerCannotBackPrediction,
    #[error("imported numerical thermal field requires result-receipt closure before admitted prediction evidence")]
    ImportedNumericalResultNotClosed,
    #[error("observed/measured thermal evidence must not use a prediction model as measurement authority")]
    MeasuredEvidenceCannotUseModelProfile,
}

fn validate_token(field: &'static str, value: &str) -> Result<(), ThermalError> {
    if value.is_empty() {
        return Err(ThermalError::EmptyField { field });
    }
    if value.trim() != value {
        return Err(ThermalError::NonCanonicalWhitespace { field });
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
pub struct ThermalNetworkId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ThermalModelProfileId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ThermalEvidenceSnapshotId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThermalMemberRoleV1 {
    HeatSource,
    ThermalMass,
    ThermalInterface,
    HeatSpreader,
    HeatSinkOrRadiator,
    ColdPlate,
    HeatPipeOrVaporTransport,
    CoolingFluidInterface,
    AmbientBoundary,
    InsulationOrBarrier,
    ThermoelectricElement,
    PhaseChangeStorage,
    TemperatureObservationLocation,
    HeatFluxObservationLocation,
    GeometryRegion,
    Other(String),
}

impl ThermalMemberRoleV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::HeatSource => "heat-source".into(),
            Self::ThermalMass => "thermal-mass".into(),
            Self::ThermalInterface => "thermal-interface".into(),
            Self::HeatSpreader => "heat-spreader".into(),
            Self::HeatSinkOrRadiator => "heat-sink-or-radiator".into(),
            Self::ColdPlate => "cold-plate".into(),
            Self::HeatPipeOrVaporTransport => "heat-pipe-or-vapor-transport".into(),
            Self::CoolingFluidInterface => "cooling-fluid-interface".into(),
            Self::AmbientBoundary => "ambient-boundary".into(),
            Self::InsulationOrBarrier => "insulation-or-barrier".into(),
            Self::ThermoelectricElement => "thermoelectric-element".into(),
            Self::PhaseChangeStorage => "phase-change-storage".into(),
            Self::TemperatureObservationLocation => "temperature-observation-location".into(),
            Self::HeatFluxObservationLocation => "heat-flux-observation-location".into(),
            Self::GeometryRegion => "geometry-region".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), ThermalError> {
        if let Self::Other(value) = self {
            validate_token("member_role.other", value)?;
        }
        Ok(())
    }

    fn component_required(&self) -> bool {
        matches!(
            self,
            Self::HeatSource
                | Self::ThermalMass
                | Self::ThermalInterface
                | Self::HeatSpreader
                | Self::HeatSinkOrRadiator
                | Self::ColdPlate
                | Self::HeatPipeOrVaporTransport
                | Self::InsulationOrBarrier
                | Self::ThermoelectricElement
                | Self::PhaseChangeStorage
        )
    }

    fn abstract_region(&self) -> bool {
        matches!(
            self,
            Self::CoolingFluidInterface
                | Self::AmbientBoundary
                | Self::TemperatureObservationLocation
                | Self::HeatFluxObservationLocation
                | Self::GeometryRegion
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThermalMemberV1 {
    pub member_id: String,
    pub role: ThermalMemberRoleV1,
    pub component_subject_id: Option<ComponentSubjectId>,
    pub geometry_binding_id: Option<String>,
    pub material_state_ref: Option<String>,
    pub surface_or_interface_state_ref: Option<String>,
    pub display_name: Option<String>,
}

impl ThermalMemberV1 {
    pub fn validate(&self) -> Result<(), ThermalError> {
        validate_token("member_id", &self.member_id)?;
        self.role.validate()?;
        if self.role.component_required() && self.component_subject_id.is_none() {
            return Err(ThermalError::ComponentRequired(self.member_id.clone()));
        }
        if self.role.abstract_region() && self.component_subject_id.is_some() {
            return Err(ThermalError::AbstractRegionCannotBeComponent(
                self.member_id.clone(),
            ));
        }
        for (field, value) in [
            ("geometry_binding_id", self.geometry_binding_id.as_deref()),
            ("material_state_ref", self.material_state_ref.as_deref()),
            (
                "surface_or_interface_state_ref",
                self.surface_or_interface_state_ref.as_deref(),
            ),
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
            self.surface_or_interface_state_ref
                .as_deref()
                .unwrap_or("none")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThermalRelationKindV1 {
    ConductivePath,
    ContactInterfacePath,
    ConvectiveBoundary,
    RadiativeExchange,
    HeatSourceCoupling,
    CoolingLoopInterfaceRef,
    PhaseChangeCoupling,
    SensorObservationRelation,
    GeometryOrMaterialBinding,
    ThermoMechanicalCouplingRef,
    Other(String),
}

impl ThermalRelationKindV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::ConductivePath => "conductive-path".into(),
            Self::ContactInterfacePath => "contact-interface-path".into(),
            Self::ConvectiveBoundary => "convective-boundary".into(),
            Self::RadiativeExchange => "radiative-exchange".into(),
            Self::HeatSourceCoupling => "heat-source-coupling".into(),
            Self::CoolingLoopInterfaceRef => "cooling-loop-interface-ref".into(),
            Self::PhaseChangeCoupling => "phase-change-coupling".into(),
            Self::SensorObservationRelation => "sensor-observation-relation".into(),
            Self::GeometryOrMaterialBinding => "geometry-or-material-binding".into(),
            Self::ThermoMechanicalCouplingRef => "thermo-mechanical-coupling-ref".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), ThermalError> {
        if let Self::Other(value) = self {
            validate_token("relation_kind.other", value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThermalRelationV1 {
    pub relation_id: String,
    pub kind: ThermalRelationKindV1,
    pub from_member_id: String,
    pub to_member_id: String,
    pub coupling_profile_id: Option<String>,
}

impl ThermalRelationV1 {
    fn validate(&self) -> Result<(), ThermalError> {
        validate_token("relation_id", &self.relation_id)?;
        validate_token("from_member_id", &self.from_member_id)?;
        validate_token("to_member_id", &self.to_member_id)?;
        self.kind.validate()?;
        if let Some(value) = self.coupling_profile_id.as_deref() {
            validate_token("coupling_profile_id", value)?;
        }
        if self.from_member_id == self.to_member_id {
            return Err(ThermalError::SelfRelation(self.relation_id.clone()));
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
pub struct ThermalNetworkV1 {
    pub network_kind_id: String,
    #[serde(default)]
    pub members: Vec<ThermalMemberV1>,
    #[serde(default)]
    pub relations: Vec<ThermalRelationV1>,
    pub display_name: Option<String>,
}

impl ThermalNetworkV1 {
    pub fn validate(&self) -> Result<(), ThermalError> {
        validate_token("network_kind_id", &self.network_kind_id)?;
        let mut members = BTreeSet::new();
        for member in &self.members {
            member.validate()?;
            if !members.insert(member.member_id.clone()) {
                return Err(ThermalError::DuplicateMember(member.member_id.clone()));
            }
        }
        let mut relations = BTreeSet::new();
        for relation in &self.relations {
            relation.validate()?;
            if !relations.insert(relation.relation_id.clone()) {
                return Err(ThermalError::DuplicateRelation(relation.relation_id.clone()));
            }
            if !members.contains(&relation.from_member_id) {
                return Err(ThermalError::UnknownMember(relation.from_member_id.clone()));
            }
            if !members.contains(&relation.to_member_id) {
                return Err(ThermalError::UnknownMember(relation.to_member_id.clone()));
            }
        }
        Ok(())
    }

    pub fn network_id(&self) -> Result<ThermalNetworkId, ThermalError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, NETWORK_DOMAIN);
        hash_field(&mut hasher, &self.network_kind_id);
        let mut members = self
            .members
            .iter()
            .map(ThermalMemberV1::canonical_key)
            .collect::<Vec<_>>();
        members.sort();
        for member in members {
            hash_field(&mut hasher, &member);
        }
        let mut relations = self
            .relations
            .iter()
            .map(ThermalRelationV1::canonical_key)
            .collect::<Vec<_>>();
        relations.sort();
        for relation in relations {
            hash_field(&mut hasher, &relation);
        }
        Ok(ThermalNetworkId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThermalModelFamilyV1 {
    LumpedThermalNetwork,
    SteadyConduction,
    TransientConduction,
    AnalyticalOrEmpiricalConvection,
    RadiationExchange,
    PhaseChangeReducedOrder,
    ExternalFiniteElementRequired,
    ExternalCfdOrConjugateHeatTransferRequired,
    CoupledElectroThermal,
    CoupledMagnetoThermal,
    CoupledThermoOptic,
    ImportedNumericalField,
    UnknownNotAdmitted,
    Other(String),
}

impl ThermalModelFamilyV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::LumpedThermalNetwork => "lumped-thermal-network".into(),
            Self::SteadyConduction => "steady-conduction".into(),
            Self::TransientConduction => "transient-conduction".into(),
            Self::AnalyticalOrEmpiricalConvection => "analytical-or-empirical-convection".into(),
            Self::RadiationExchange => "radiation-exchange".into(),
            Self::PhaseChangeReducedOrder => "phase-change-reduced-order".into(),
            Self::ExternalFiniteElementRequired => "external-finite-element-required".into(),
            Self::ExternalCfdOrConjugateHeatTransferRequired => {
                "external-cfd-or-conjugate-heat-transfer-required".into()
            }
            Self::CoupledElectroThermal => "coupled-electro-thermal".into(),
            Self::CoupledMagnetoThermal => "coupled-magneto-thermal".into(),
            Self::CoupledThermoOptic => "coupled-thermo-optic".into(),
            Self::ImportedNumericalField => "imported-numerical-field".into(),
            Self::UnknownNotAdmitted => "unknown-not-admitted".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), ThermalError> {
        if let Self::Other(value) = self {
            validate_token("model_family.other", value)?;
        }
        Ok(())
    }

    pub fn is_backend_requirement(&self) -> bool {
        matches!(
            self,
            Self::ExternalFiniteElementRequired | Self::ExternalCfdOrConjugateHeatTransferRequired
        )
    }

    pub fn supports_prediction(&self) -> bool {
        !matches!(
            self,
            Self::UnknownNotAdmitted
                | Self::ExternalFiniteElementRequired
                | Self::ExternalCfdOrConjugateHeatTransferRequired
                | Self::ImportedNumericalField
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThermalModelProfileV1 {
    pub network_id: ThermalNetworkId,
    pub model_family: ThermalModelFamilyV1,
    pub applicability_profile_id: String,
    #[serde(default)]
    pub source_evidence_refs: Vec<String>,
    #[serde(default)]
    pub boundary_profile_refs: Vec<String>,
    #[serde(default)]
    pub geometry_binding_refs: Vec<String>,
    #[serde(default)]
    pub material_evidence_refs: Vec<String>,
}

impl ThermalModelProfileV1 {
    pub fn validate_for(&self, network: &ThermalNetworkV1) -> Result<(), ThermalError> {
        self.model_family.validate()?;
        validate_token("applicability_profile_id", &self.applicability_profile_id)?;
        if self.network_id != network.network_id()? {
            return Err(ThermalError::NetworkMismatch);
        }
        if self.model_family == ThermalModelFamilyV1::UnknownNotAdmitted {
            return Err(ThermalError::UnknownModelNotPredictive);
        }
        if self.source_evidence_refs.is_empty() {
            return Err(ThermalError::PredictiveEvidenceRequired);
        }
        for (field, values) in [
            ("source_evidence_ref", &self.source_evidence_refs),
            ("boundary_profile_ref", &self.boundary_profile_refs),
            ("geometry_binding_ref", &self.geometry_binding_refs),
            ("material_evidence_ref", &self.material_evidence_refs),
        ] {
            for value in values {
                validate_token(field, value)?;
            }
        }
        Ok(())
    }

    pub fn model_profile_id(
        &self,
        network: &ThermalNetworkV1,
    ) -> Result<ThermalModelProfileId, ThermalError> {
        self.validate_for(network)?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, MODEL_DOMAIN);
        hash_field(&mut hasher, &self.network_id.0);
        hash_field(&mut hasher, &self.model_family.canonical_tag());
        hash_field(&mut hasher, &self.applicability_profile_id);
        for mut values in [
            self.source_evidence_refs.clone(),
            self.boundary_profile_refs.clone(),
            self.geometry_binding_refs.clone(),
            self.material_evidence_refs.clone(),
        ] {
            values.sort();
            for value in values {
                hash_field(&mut hasher, &value);
            }
            hasher.update(&[0xff]);
        }
        Ok(ThermalModelProfileId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThermalEvidencePlaneV1 {
    DeclaredRated,
    ConfiguredCommanded,
    Predicted,
    InferredDerived,
    ObservedMeasured,
}

impl ThermalEvidencePlaneV1 {
    pub fn satisfies_exact(self, required: Self) -> Result<(), ThermalError> {
        if self == required {
            Ok(())
        } else {
            Err(ThermalError::EvidencePlaneMismatch)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThermalEvidenceRefV1 {
    pub evidence_id: String,
    pub network_id: ThermalNetworkId,
    pub plane: ThermalEvidencePlaneV1,
    pub semantic_kind_id: String,
    pub member_id: Option<String>,
    pub model_profile_id: Option<ThermalModelProfileId>,
}

impl ThermalEvidenceRefV1 {
    pub fn validate_for(&self, network: &ThermalNetworkV1) -> Result<(), ThermalError> {
        validate_token("evidence_id", &self.evidence_id)?;
        validate_token("semantic_kind_id", &self.semantic_kind_id)?;
        if self.network_id != network.network_id()? {
            return Err(ThermalError::EvidenceNetworkMismatch);
        }
        if let Some(member_id) = self.member_id.as_deref() {
            validate_token("member_id", member_id)?;
            if !network.members.iter().any(|member| member.member_id == member_id) {
                return Err(ThermalError::EvidenceUnknownMember(member_id.into()));
            }
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        format!(
            "{}|{}|{:?}|{}|{}|{}",
            self.evidence_id,
            self.network_id.0,
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
pub struct ThermalEvidenceSnapshotV1 {
    pub network: ThermalNetworkV1,
    #[serde(default)]
    pub model_profiles: Vec<ThermalModelProfileV1>,
    #[serde(default)]
    pub evidence_refs: Vec<ThermalEvidenceRefV1>,
}

impl ThermalEvidenceSnapshotV1 {
    pub fn snapshot_id(&self) -> Result<ThermalEvidenceSnapshotId, ThermalError> {
        let network_id = self.network.network_id()?;
        let mut models = BTreeMap::new();
        for model in &self.model_profiles {
            let id = model.model_profile_id(&self.network)?;
            if models.insert(id.0.clone(), model).is_some() {
                return Err(ThermalError::DuplicateModelProfile(id.0));
            }
        }

        let mut evidence_ids = BTreeSet::new();
        let mut evidence = Vec::new();
        for item in &self.evidence_refs {
            item.validate_for(&self.network)?;
            if !evidence_ids.insert(item.evidence_id.clone()) {
                return Err(ThermalError::DuplicateEvidence(item.evidence_id.clone()));
            }

            let model = match &item.model_profile_id {
                Some(id) => Some(
                    models
                        .get(&id.0)
                        .copied()
                        .ok_or_else(|| ThermalError::UnknownModelProfile(id.0.clone()))?,
                ),
                None => None,
            };

            match item.plane {
                ThermalEvidencePlaneV1::Predicted => {
                    let model = model.ok_or(ThermalError::PredictedEvidenceRequiresModelProfile)?;
                    if model.model_family.is_backend_requirement() {
                        return Err(ThermalError::RequirementMarkerCannotBackPrediction);
                    }
                    if model.model_family == ThermalModelFamilyV1::ImportedNumericalField {
                        return Err(ThermalError::ImportedNumericalResultNotClosed);
                    }
                    if !model.model_family.supports_prediction() {
                        return Err(ThermalError::RequirementMarkerCannotBackPrediction);
                    }
                }
                ThermalEvidencePlaneV1::ObservedMeasured if model.is_some() => {
                    return Err(ThermalError::MeasuredEvidenceCannotUseModelProfile);
                }
                _ => {}
            }
            evidence.push(item.canonical_key());
        }
        evidence.sort();

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, SNAPSHOT_DOMAIN);
        hash_field(&mut hasher, &network_id.0);
        for id in models.keys() {
            hash_field(&mut hasher, id);
        }
        for item in evidence {
            hash_field(&mut hasher, &item);
        }
        Ok(ThermalEvidenceSnapshotId(
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

    fn network() -> ThermalNetworkV1 {
        ThermalNetworkV1 {
            network_kind_id: "thermal-reference".into(),
            members: vec![
                ThermalMemberV1 {
                    member_id: "device".into(),
                    role: ThermalMemberRoleV1::HeatSource,
                    component_subject_id: Some(cid('a')),
                    geometry_binding_id: Some("geom:device".into()),
                    material_state_ref: Some("mat:device".into()),
                    surface_or_interface_state_ref: None,
                    display_name: Some("Device".into()),
                },
                ThermalMemberV1 {
                    member_id: "sink".into(),
                    role: ThermalMemberRoleV1::HeatSinkOrRadiator,
                    component_subject_id: Some(cid('b')),
                    geometry_binding_id: Some("geom:sink".into()),
                    material_state_ref: Some("mat:sink".into()),
                    surface_or_interface_state_ref: Some("surface:installed".into()),
                    display_name: Some("Sink".into()),
                },
                ThermalMemberV1 {
                    member_id: "ambient".into(),
                    role: ThermalMemberRoleV1::AmbientBoundary,
                    component_subject_id: None,
                    geometry_binding_id: None,
                    material_state_ref: None,
                    surface_or_interface_state_ref: None,
                    display_name: Some("Ambient".into()),
                },
            ],
            relations: vec![
                ThermalRelationV1 {
                    relation_id: "device-sink".into(),
                    kind: ThermalRelationKindV1::ConductivePath,
                    from_member_id: "device".into(),
                    to_member_id: "sink".into(),
                    coupling_profile_id: Some("path:v1".into()),
                },
                ThermalRelationV1 {
                    relation_id: "sink-ambient".into(),
                    kind: ThermalRelationKindV1::ConvectiveBoundary,
                    from_member_id: "sink".into(),
                    to_member_id: "ambient".into(),
                    coupling_profile_id: Some("boundary:v1".into()),
                },
            ],
            display_name: Some("Reference".into()),
        }
    }

    fn profile(network: &ThermalNetworkV1, family: ThermalModelFamilyV1) -> ThermalModelProfileV1 {
        ThermalModelProfileV1 {
            network_id: network.network_id().unwrap(),
            model_family: family,
            applicability_profile_id: "fixture:v1".into(),
            source_evidence_refs: vec!["source:model".into()],
            boundary_profile_refs: vec!["boundary:fixture".into()],
            geometry_binding_refs: vec![],
            material_evidence_refs: vec![],
        }
    }

    #[test]
    fn labels_do_not_change_identity_but_path_changes_do() {
        let a = network();
        let mut b = a.clone();
        b.display_name = Some("Renamed".into());
        assert_eq!(a.network_id().unwrap(), b.network_id().unwrap());
        b.relations[0].coupling_profile_id = Some("path:v2".into());
        assert_ne!(a.network_id().unwrap(), b.network_id().unwrap());
    }

    #[test]
    fn fem_and_cfd_requirements_cannot_mint_predictions() {
        for family in [
            ThermalModelFamilyV1::ExternalFiniteElementRequired,
            ThermalModelFamilyV1::ExternalCfdOrConjugateHeatTransferRequired,
        ] {
            let network = network();
            let model = profile(&network, family);
            let model_id = model.model_profile_id(&network).unwrap();
            let snapshot = ThermalEvidenceSnapshotV1 {
                network: network.clone(),
                model_profiles: vec![model],
                evidence_refs: vec![ThermalEvidenceRefV1 {
                    evidence_id: "prediction:false".into(),
                    network_id: network.network_id().unwrap(),
                    plane: ThermalEvidencePlaneV1::Predicted,
                    semantic_kind_id: "thermal.temperature".into(),
                    member_id: Some("device".into()),
                    model_profile_id: Some(model_id),
                }],
            };
            assert_eq!(
                snapshot.snapshot_id(),
                Err(ThermalError::RequirementMarkerCannotBackPrediction)
            );
        }
    }

    #[test]
    fn dangling_and_duplicate_model_profiles_fail_closed() {
        let network = network();
        let dangling = ThermalEvidenceSnapshotV1 {
            network: network.clone(),
            model_profiles: vec![],
            evidence_refs: vec![ThermalEvidenceRefV1 {
                evidence_id: "prediction:dangling".into(),
                network_id: network.network_id().unwrap(),
                plane: ThermalEvidencePlaneV1::Predicted,
                semantic_kind_id: "thermal.temperature".into(),
                member_id: Some("device".into()),
                model_profile_id: Some(ThermalModelProfileId("missing".into())),
            }],
        };
        assert!(matches!(
            dangling.snapshot_id(),
            Err(ThermalError::UnknownModelProfile(_))
        ));

        let model = profile(&network, ThermalModelFamilyV1::LumpedThermalNetwork);
        let duplicate = ThermalEvidenceSnapshotV1 {
            network,
            model_profiles: vec![model.clone(), model],
            evidence_refs: vec![],
        };
        assert!(matches!(
            duplicate.snapshot_id(),
            Err(ThermalError::DuplicateModelProfile(_))
        ));
    }

    #[test]
    fn imported_numerical_field_requires_result_closure() {
        let network = network();
        let model = profile(&network, ThermalModelFamilyV1::ImportedNumericalField);
        let model_id = model.model_profile_id(&network).unwrap();
        let snapshot = ThermalEvidenceSnapshotV1 {
            network: network.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![ThermalEvidenceRefV1 {
                evidence_id: "prediction:imported".into(),
                network_id: network.network_id().unwrap(),
                plane: ThermalEvidencePlaneV1::Predicted,
                semantic_kind_id: "thermal.field".into(),
                member_id: Some("device".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(ThermalError::ImportedNumericalResultNotClosed)
        );
    }

    #[test]
    fn measured_temperature_cannot_use_prediction_model_as_authority() {
        let network = network();
        let model = profile(&network, ThermalModelFamilyV1::LumpedThermalNetwork);
        let model_id = model.model_profile_id(&network).unwrap();
        let snapshot = ThermalEvidenceSnapshotV1 {
            network: network.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![ThermalEvidenceRefV1 {
                evidence_id: "measurement:temperature".into(),
                network_id: network.network_id().unwrap(),
                plane: ThermalEvidencePlaneV1::ObservedMeasured,
                semantic_kind_id: "thermal.temperature".into(),
                member_id: Some("device".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert_eq!(
            snapshot.snapshot_id(),
            Err(ThermalError::MeasuredEvidenceCannotUseModelProfile)
        );
    }

    #[test]
    fn valid_lumped_prediction_remains_prediction_only() {
        let network = network();
        let model = profile(&network, ThermalModelFamilyV1::LumpedThermalNetwork);
        let model_id = model.model_profile_id(&network).unwrap();
        let snapshot = ThermalEvidenceSnapshotV1 {
            network: network.clone(),
            model_profiles: vec![model],
            evidence_refs: vec![ThermalEvidenceRefV1 {
                evidence_id: "prediction:temperature".into(),
                network_id: network.network_id().unwrap(),
                plane: ThermalEvidencePlaneV1::Predicted,
                semantic_kind_id: "thermal.temperature".into(),
                member_id: Some("device".into()),
                model_profile_id: Some(model_id),
            }],
        };
        assert!(snapshot.snapshot_id().is_ok());
        assert_ne!(
            ThermalEvidencePlaneV1::Predicted,
            ThermalEvidencePlaneV1::ObservedMeasured
        );
    }

    #[test]
    fn serialization_does_not_bypass_dangling_reference_validation() {
        let network = network();
        let snapshot = ThermalEvidenceSnapshotV1 {
            network: network.clone(),
            model_profiles: vec![],
            evidence_refs: vec![ThermalEvidenceRefV1 {
                evidence_id: "prediction:dangling".into(),
                network_id: network.network_id().unwrap(),
                plane: ThermalEvidencePlaneV1::Predicted,
                semantic_kind_id: "thermal.temperature".into(),
                member_id: None,
                model_profile_id: Some(ThermalModelProfileId("missing".into())),
            }],
        };
        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: ThermalEvidenceSnapshotV1 = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.snapshot_id().is_err());
    }
}
