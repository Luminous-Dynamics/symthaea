// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed request semantics for the bounded DEVSIM semiconductor reference path.
//!
//! REF-001C-1 owns request validation and deterministic identity only. It does
//! not render solver input, spawn Python/DEVSIM, parse output, or establish any
//! physical-device claim.
//!
//! ```text
//! typed request accepted
//! != DEVSIM input rendered
//! != solver executed
//! != mesh qualified
//! != semiconductor model validated
//! != physical device validated
//! ```

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const DEVSIM_DIODE_REQUEST_SCHEMA_V1: &str = "eng-semi-ref-001c-devsim-diode-request-v1";
pub const MAX_ABS_BIAS_VOLTS_V1: f64 = 1.0;
pub const MAX_BIAS_POINTS_V1: usize = 1_001;
pub const MAX_ESTIMATED_MESH_INTERVALS_V1: f64 = 1_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionRole {
    PType,
    NType,
}

impl RegionRole {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::PType => "p_type",
            Self::NType => "n_type",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemiconductorMaterial {
    Silicon,
}

impl SemiconductorMaterial {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Silicon => "silicon",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactRole {
    Anode,
    Cathode,
}

impl ContactRole {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Anode => "anode",
            Self::Cathode => "cathode",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevsimPhysicsProfile {
    SiliconDriftDiffusionV1,
}

impl DevsimPhysicsProfile {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::SiliconDriftDiffusionV1 => "silicon_drift_diffusion_v1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestedObservable {
    AnodeCurrent,
    CathodeCurrent,
    ElectrostaticPotential,
    ElectronDensity,
    HoleDensity,
}

impl RequestedObservable {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::AnodeCurrent => "anode_current",
            Self::CathodeCurrent => "cathode_current",
            Self::ElectrostaticPotential => "electrostatic_potential",
            Self::ElectronDensity => "electron_density",
            Self::HoleDensity => "hole_density",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry1d {
    pub total_length_m: f64,
    pub junction_position_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region1d {
    pub id: String,
    pub role: RegionRole,
    pub start_m: f64,
    pub end_m: f64,
    pub material: SemiconductorMaterial,
    /// Explicit non-negative donor magnitude in m^-3. V1 never overloads sign.
    pub donor_density_per_m3: f64,
    /// Explicit non-negative acceptor magnitude in m^-3. V1 never overloads sign.
    pub acceptor_density_per_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact1d {
    pub id: String,
    pub role: ContactRole,
    pub position_m: f64,
    pub region_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshPolicy1d {
    pub p_bulk_spacing_m: f64,
    pub junction_spacing_m: f64,
    pub n_bulk_spacing_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiasSweepV1 {
    /// Anode voltage relative to a fixed 0 V cathode.
    pub start_volts: f64,
    pub stop_volts: f64,
    pub step_volts: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverSettingsV1 {
    pub max_dc_iterations: u32,
    pub relative_error: f64,
    pub absolute_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevsimDiodeRequestV1 {
    pub schema_id: String,
    pub subject_ref: String,
    pub analytical_profile_ref: String,
    pub device_id: String,
    pub geometry: Geometry1d,
    pub p_region: Region1d,
    pub n_region: Region1d,
    pub anode: Contact1d,
    pub cathode: Contact1d,
    pub mesh_policy: MeshPolicy1d,
    pub physics_profile: DevsimPhysicsProfile,
    pub temperature_kelvin: f64,
    pub bias_sweep: BiasSweepV1,
    pub solver_settings: SolverSettingsV1,
    pub requested_observables: Vec<RequestedObservable>,
}

impl DevsimDiodeRequestV1 {
    pub fn validate(&self) -> Result<(), RequestError> {
        if self.schema_id != DEVSIM_DIODE_REQUEST_SCHEMA_V1 {
            return Err(RequestError::UnsupportedSchema(self.schema_id.clone()));
        }
        validate_id("subject_ref", &self.subject_ref)?;
        validate_id("analytical_profile_ref", &self.analytical_profile_ref)?;
        validate_id("device_id", &self.device_id)?;

        validate_finite("total_length_m", self.geometry.total_length_m)?;
        validate_finite("junction_position_m", self.geometry.junction_position_m)?;
        if self.geometry.total_length_m <= 0.0 {
            return Err(RequestError::NonPositiveGeometryLength);
        }
        if self.geometry.junction_position_m <= 0.0
            || self.geometry.junction_position_m >= self.geometry.total_length_m
        {
            return Err(RequestError::JunctionOutsideDevice);
        }

        self.validate_region(&self.p_region)?;
        self.validate_region(&self.n_region)?;
        if self.p_region.id == self.n_region.id {
            return Err(RequestError::DuplicateRegionId(self.p_region.id.clone()));
        }
        if self.p_region.role != RegionRole::PType || self.n_region.role != RegionRole::NType {
            return Err(RequestError::RegionRoleMismatch);
        }
        if self.p_region.start_m != 0.0
            || self.p_region.end_m != self.geometry.junction_position_m
            || self.n_region.start_m != self.geometry.junction_position_m
            || self.n_region.end_m != self.geometry.total_length_m
        {
            return Err(RequestError::RegionBoundaryMismatch);
        }
        if self.p_region.material != SemiconductorMaterial::Silicon
            || self.n_region.material != SemiconductorMaterial::Silicon
        {
            return Err(RequestError::UnsupportedMaterialProfile);
        }
        if self.p_region.acceptor_density_per_m3 <= 0.0
            || self.p_region.donor_density_per_m3 != 0.0
            || self.n_region.donor_density_per_m3 <= 0.0
            || self.n_region.acceptor_density_per_m3 != 0.0
        {
            return Err(RequestError::AmbiguousV1DopingProfile);
        }

        self.validate_contact(&self.anode)?;
        self.validate_contact(&self.cathode)?;
        if self.anode.id == self.cathode.id {
            return Err(RequestError::DuplicateContactId(self.anode.id.clone()));
        }
        if self.anode.role != ContactRole::Anode
            || self.anode.position_m != 0.0
            || self.anode.region_id != self.p_region.id
        {
            return Err(RequestError::AnodePlacementMismatch);
        }
        if self.cathode.role != ContactRole::Cathode
            || self.cathode.position_m != self.geometry.total_length_m
            || self.cathode.region_id != self.n_region.id
        {
            return Err(RequestError::CathodePlacementMismatch);
        }

        self.validate_mesh()?;
        self.validate_temperature()?;
        self.validate_bias_sweep()?;
        self.validate_solver_settings()?;
        self.validate_observables()?;
        Ok(())
    }

    pub fn request_id(&self) -> Result<String, RequestError> {
        self.validate()?;
        let mut bytes = Vec::new();
        push_text(&mut bytes, "schema", &self.schema_id);
        push_text(&mut bytes, "subject_ref", &self.subject_ref);
        push_text(
            &mut bytes,
            "analytical_profile_ref",
            &self.analytical_profile_ref,
        );
        push_text(&mut bytes, "device_id", &self.device_id);

        push_f64(&mut bytes, "geometry.total_length_m", self.geometry.total_length_m);
        push_f64(
            &mut bytes,
            "geometry.junction_position_m",
            self.geometry.junction_position_m,
        );
        push_region(&mut bytes, "p_region", &self.p_region);
        push_region(&mut bytes, "n_region", &self.n_region);
        push_contact(&mut bytes, "anode", &self.anode);
        push_contact(&mut bytes, "cathode", &self.cathode);

        push_f64(
            &mut bytes,
            "mesh.p_bulk_spacing_m",
            self.mesh_policy.p_bulk_spacing_m,
        );
        push_f64(
            &mut bytes,
            "mesh.junction_spacing_m",
            self.mesh_policy.junction_spacing_m,
        );
        push_f64(
            &mut bytes,
            "mesh.n_bulk_spacing_m",
            self.mesh_policy.n_bulk_spacing_m,
        );
        push_text(
            &mut bytes,
            "physics_profile",
            self.physics_profile.canonical_name(),
        );
        push_f64(&mut bytes, "temperature_kelvin", self.temperature_kelvin);
        push_f64(&mut bytes, "bias.start_volts", self.bias_sweep.start_volts);
        push_f64(&mut bytes, "bias.stop_volts", self.bias_sweep.stop_volts);
        push_f64(&mut bytes, "bias.step_volts", self.bias_sweep.step_volts);
        push_u32(
            &mut bytes,
            "solver.max_dc_iterations",
            self.solver_settings.max_dc_iterations,
        );
        push_f64(
            &mut bytes,
            "solver.relative_error",
            self.solver_settings.relative_error,
        );
        push_f64(
            &mut bytes,
            "solver.absolute_error",
            self.solver_settings.absolute_error,
        );

        let mut observables = self.requested_observables.clone();
        observables.sort_unstable();
        for observable in observables {
            push_text(&mut bytes, "observable", observable.canonical_name());
        }

        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }

    fn validate_region(&self, region: &Region1d) -> Result<(), RequestError> {
        validate_id("region.id", &region.id)?;
        validate_finite("region.start_m", region.start_m)?;
        validate_finite("region.end_m", region.end_m)?;
        if region.start_m < 0.0 || region.end_m <= region.start_m {
            return Err(RequestError::InvalidRegionExtent(region.id.clone()));
        }
        validate_density(&region.id, "donor", region.donor_density_per_m3)?;
        validate_density(&region.id, "acceptor", region.acceptor_density_per_m3)?;
        Ok(())
    }

    fn validate_contact(&self, contact: &Contact1d) -> Result<(), RequestError> {
        validate_id("contact.id", &contact.id)?;
        validate_id("contact.region_id", &contact.region_id)?;
        validate_finite("contact.position_m", contact.position_m)?;
        if contact.position_m < 0.0 || contact.position_m > self.geometry.total_length_m {
            return Err(RequestError::ContactOutsideDevice(contact.id.clone()));
        }
        Ok(())
    }

    fn validate_mesh(&self) -> Result<(), RequestError> {
        for (name, spacing) in [
            ("p_bulk_spacing_m", self.mesh_policy.p_bulk_spacing_m),
            ("junction_spacing_m", self.mesh_policy.junction_spacing_m),
            ("n_bulk_spacing_m", self.mesh_policy.n_bulk_spacing_m),
        ] {
            validate_finite(name, spacing)?;
            if spacing <= 0.0 {
                return Err(RequestError::NonPositiveMeshSpacing(name));
            }
            if spacing > self.geometry.total_length_m {
                return Err(RequestError::MeshSpacingExceedsDevice(name));
            }
        }
        if self.mesh_policy.junction_spacing_m > self.mesh_policy.p_bulk_spacing_m
            || self.mesh_policy.junction_spacing_m > self.mesh_policy.n_bulk_spacing_m
        {
            return Err(RequestError::JunctionMeshNotRefined);
        }
        let minimum_spacing = self
            .mesh_policy
            .p_bulk_spacing_m
            .min(self.mesh_policy.junction_spacing_m)
            .min(self.mesh_policy.n_bulk_spacing_m);
        let estimated_intervals = self.geometry.total_length_m / minimum_spacing;
        if estimated_intervals > MAX_ESTIMATED_MESH_INTERVALS_V1 {
            return Err(RequestError::MeshBudgetExceeded);
        }
        Ok(())
    }

    fn validate_temperature(&self) -> Result<(), RequestError> {
        validate_finite("temperature_kelvin", self.temperature_kelvin)?;
        if self.temperature_kelvin <= 0.0 {
            return Err(RequestError::NonPositiveTemperature);
        }
        Ok(())
    }

    fn validate_bias_sweep(&self) -> Result<(), RequestError> {
        validate_finite("bias.start_volts", self.bias_sweep.start_volts)?;
        validate_finite("bias.stop_volts", self.bias_sweep.stop_volts)?;
        validate_finite("bias.step_volts", self.bias_sweep.step_volts)?;
        if self.bias_sweep.start_volts > self.bias_sweep.stop_volts {
            return Err(RequestError::ReversedBiasSweep);
        }
        if self.bias_sweep.step_volts <= 0.0 {
            return Err(RequestError::NonPositiveBiasStep);
        }
        if self.bias_sweep.start_volts.abs() > MAX_ABS_BIAS_VOLTS_V1
            || self.bias_sweep.stop_volts.abs() > MAX_ABS_BIAS_VOLTS_V1
        {
            return Err(RequestError::BiasOutsideV1Envelope);
        }

        let intervals =
            (self.bias_sweep.stop_volts - self.bias_sweep.start_volts) / self.bias_sweep.step_volts;
        let nearest = intervals.round();
        let scale = intervals.abs().max(1.0);
        if (intervals - nearest).abs() > 1.0e-9 * scale {
            return Err(RequestError::BiasStepDoesNotLandOnStop);
        }
        let points = nearest as usize + 1;
        if points > MAX_BIAS_POINTS_V1 {
            return Err(RequestError::BiasPointBudgetExceeded(points));
        }
        Ok(())
    }

    fn validate_solver_settings(&self) -> Result<(), RequestError> {
        if self.solver_settings.max_dc_iterations == 0 {
            return Err(RequestError::ZeroIterationBudget);
        }
        for (name, value) in [
            ("solver.relative_error", self.solver_settings.relative_error),
            ("solver.absolute_error", self.solver_settings.absolute_error),
        ] {
            validate_finite(name, value)?;
            if value <= 0.0 {
                return Err(RequestError::NonPositiveSolverTolerance(name));
            }
        }
        Ok(())
    }

    fn validate_observables(&self) -> Result<(), RequestError> {
        if self.requested_observables.is_empty() {
            return Err(RequestError::NoRequestedObservables);
        }
        let mut seen = BTreeSet::new();
        for observable in &self.requested_observables {
            if !seen.insert(*observable) {
                return Err(RequestError::DuplicateObservable(*observable));
            }
        }
        Ok(())
    }
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), RequestError> {
    if value.trim().is_empty() {
        return Err(RequestError::EmptyIdentifier(kind));
    }
    if value.trim() != value {
        return Err(RequestError::NonCanonicalIdentifier {
            kind,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn validate_finite(kind: &'static str, value: f64) -> Result<(), RequestError> {
    if !value.is_finite() {
        return Err(RequestError::NonFiniteValue(kind));
    }
    Ok(())
}

fn validate_density(
    region: &str,
    carrier: &'static str,
    value: f64,
) -> Result<(), RequestError> {
    if !value.is_finite() {
        return Err(RequestError::NonFiniteDoping {
            region: region.to_string(),
            carrier,
        });
    }
    if value < 0.0 {
        return Err(RequestError::NegativeDopingMagnitude {
            region: region.to_string(),
            carrier,
        });
    }
    Ok(())
}

fn push_region(bytes: &mut Vec<u8>, prefix: &str, region: &Region1d) {
    push_text(bytes, &format!("{prefix}.id"), &region.id);
    push_text(bytes, &format!("{prefix}.role"), region.role.canonical_name());
    push_f64(bytes, &format!("{prefix}.start_m"), region.start_m);
    push_f64(bytes, &format!("{prefix}.end_m"), region.end_m);
    push_text(
        bytes,
        &format!("{prefix}.material"),
        region.material.canonical_name(),
    );
    push_f64(
        bytes,
        &format!("{prefix}.donor_density_per_m3"),
        region.donor_density_per_m3,
    );
    push_f64(
        bytes,
        &format!("{prefix}.acceptor_density_per_m3"),
        region.acceptor_density_per_m3,
    );
}

fn push_contact(bytes: &mut Vec<u8>, prefix: &str, contact: &Contact1d) {
    push_text(bytes, &format!("{prefix}.id"), &contact.id);
    push_text(
        bytes,
        &format!("{prefix}.role"),
        contact.role.canonical_name(),
    );
    push_f64(bytes, &format!("{prefix}.position_m"), contact.position_m);
    push_text(
        bytes,
        &format!("{prefix}.region_id"),
        &contact.region_id,
    );
}

fn push_text(bytes: &mut Vec<u8>, name: &str, value: &str) {
    push_raw(bytes, name.as_bytes());
    push_raw(bytes, value.as_bytes());
}

fn push_f64(bytes: &mut Vec<u8>, name: &str, value: f64) {
    push_raw(bytes, name.as_bytes());
    push_raw(bytes, &canonical_bits(value).to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, name: &str, value: u32) {
    push_raw(bytes, name.as_bytes());
    push_raw(bytes, &value.to_le_bytes());
}

fn push_raw(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum RequestError {
    #[error("unsupported DEVSIM diode request schema {0:?}")]
    UnsupportedSchema(String),
    #[error("{0} identifier cannot be empty")]
    EmptyIdentifier(&'static str),
    #[error("{kind} identifier is not canonical: {value:?}")]
    NonCanonicalIdentifier { kind: &'static str, value: String },
    #[error("{0} must be finite")]
    NonFiniteValue(&'static str),
    #[error("device length must be > 0")]
    NonPositiveGeometryLength,
    #[error("junction must lie strictly inside the 1D device")]
    JunctionOutsideDevice,
    #[error("region {0:?} has an invalid extent")]
    InvalidRegionExtent(String),
    #[error("p/n region identifiers must be distinct; duplicate {0:?}")]
    DuplicateRegionId(String),
    #[error("V1 requires explicit p-type then n-type region roles")]
    RegionRoleMismatch,
    #[error("V1 region boundaries must exactly partition the declared geometry")]
    RegionBoundaryMismatch,
    #[error("V1 supports only silicon material assignments")]
    UnsupportedMaterialProfile,
    #[error("{carrier} doping in region {region:?} must be finite")]
    NonFiniteDoping {
        region: String,
        carrier: &'static str,
    },
    #[error("{carrier} doping magnitude in region {region:?} cannot be negative")]
    NegativeDopingMagnitude {
        region: String,
        carrier: &'static str,
    },
    #[error("V1 requires p-region acceptors only and n-region donors only")]
    AmbiguousV1DopingProfile,
    #[error("contact {0:?} lies outside the declared device")]
    ContactOutsideDevice(String),
    #[error("contact identifiers must be distinct; duplicate {0:?}")]
    DuplicateContactId(String),
    #[error("anode must be the left boundary contact attached to the p region")]
    AnodePlacementMismatch,
    #[error("cathode must be the right boundary contact attached to the n region")]
    CathodePlacementMismatch,
    #[error("mesh spacing {0} must be > 0")]
    NonPositiveMeshSpacing(&'static str),
    #[error("mesh spacing {0} exceeds the device length")]
    MeshSpacingExceedsDevice(&'static str),
    #[error("junction spacing must be no coarser than either bulk spacing")]
    JunctionMeshNotRefined,
    #[error("estimated V1 mesh interval budget exceeded")]
    MeshBudgetExceeded,
    #[error("absolute temperature must be > 0 K")]
    NonPositiveTemperature,
    #[error("bias sweep start must not exceed stop")]
    ReversedBiasSweep,
    #[error("bias sweep step must be > 0 V")]
    NonPositiveBiasStep,
    #[error("bias sweep exceeds the bounded +/-1 V V1 envelope")]
    BiasOutsideV1Envelope,
    #[error("bias step must land on the declared stop voltage")]
    BiasStepDoesNotLandOnStop,
    #[error("bias sweep requests {0} points, exceeding the V1 budget")]
    BiasPointBudgetExceeded(usize),
    #[error("solver iteration budget must be > 0")]
    ZeroIterationBudget,
    #[error("solver tolerance {0} must be > 0")]
    NonPositiveSolverTolerance(&'static str),
    #[error("at least one observable must be requested")]
    NoRequestedObservables,
    #[error("requested observable is duplicated: {0:?}")]
    DuplicateObservable(RequestedObservable),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> DevsimDiodeRequestV1 {
        DevsimDiodeRequestV1 {
            schema_id: DEVSIM_DIODE_REQUEST_SCHEMA_V1.into(),
            subject_ref: "eng-semi-ref-001-v1".into(),
            analytical_profile_ref: "ref-001b-p300-n1-is1e-12".into(),
            device_id: "synthetic-pn-001".into(),
            geometry: Geometry1d {
                total_length_m: 1.0e-6,
                junction_position_m: 5.0e-7,
            },
            p_region: Region1d {
                id: "p-region".into(),
                role: RegionRole::PType,
                start_m: 0.0,
                end_m: 5.0e-7,
                material: SemiconductorMaterial::Silicon,
                donor_density_per_m3: 0.0,
                acceptor_density_per_m3: 1.0e22,
            },
            n_region: Region1d {
                id: "n-region".into(),
                role: RegionRole::NType,
                start_m: 5.0e-7,
                end_m: 1.0e-6,
                material: SemiconductorMaterial::Silicon,
                donor_density_per_m3: 1.0e22,
                acceptor_density_per_m3: 0.0,
            },
            anode: Contact1d {
                id: "anode".into(),
                role: ContactRole::Anode,
                position_m: 0.0,
                region_id: "p-region".into(),
            },
            cathode: Contact1d {
                id: "cathode".into(),
                role: ContactRole::Cathode,
                position_m: 1.0e-6,
                region_id: "n-region".into(),
            },
            mesh_policy: MeshPolicy1d {
                p_bulk_spacing_m: 1.0e-8,
                junction_spacing_m: 2.0e-9,
                n_bulk_spacing_m: 1.0e-8,
            },
            physics_profile: DevsimPhysicsProfile::SiliconDriftDiffusionV1,
            temperature_kelvin: 300.0,
            bias_sweep: BiasSweepV1 {
                start_volts: -0.2,
                stop_volts: 0.8,
                step_volts: 0.1,
            },
            solver_settings: SolverSettingsV1 {
                max_dc_iterations: 50,
                relative_error: 1.0e-10,
                absolute_error: 1.0e-12,
            },
            requested_observables: vec![
                RequestedObservable::AnodeCurrent,
                RequestedObservable::CathodeCurrent,
                RequestedObservable::ElectrostaticPotential,
            ],
        }
    }

    #[test]
    fn bounded_fixture_is_valid() {
        assert_eq!(fixture().validate(), Ok(()));
    }

    #[test]
    fn exact_request_identity_repeats() {
        let request = fixture();
        assert_eq!(request.request_id().unwrap(), request.request_id().unwrap());
    }

    #[test]
    fn claim_relevant_changes_change_identity() {
        let base = fixture();
        let base_id = base.request_id().unwrap();

        let mut temperature = fixture();
        temperature.temperature_kelvin = 301.0;
        assert_ne!(base_id, temperature.request_id().unwrap());

        let mut mesh = fixture();
        mesh.mesh_policy.junction_spacing_m = 1.0e-9;
        assert_ne!(base_id, mesh.request_id().unwrap());

        let mut doping = fixture();
        doping.p_region.acceptor_density_per_m3 = 1.1e22;
        assert_ne!(base_id, doping.request_id().unwrap());
    }

    #[test]
    fn observable_order_is_non_semantic() {
        let a = fixture();
        let mut b = fixture();
        b.requested_observables.reverse();
        assert_eq!(a.request_id().unwrap(), b.request_id().unwrap());
    }

    #[test]
    fn duplicate_observable_is_rejected() {
        let mut request = fixture();
        request
            .requested_observables
            .push(RequestedObservable::AnodeCurrent);
        assert_eq!(
            request.validate(),
            Err(RequestError::DuplicateObservable(
                RequestedObservable::AnodeCurrent
            ))
        );
    }

    #[test]
    fn negative_doping_magnitude_is_rejected() {
        let mut request = fixture();
        request.p_region.acceptor_density_per_m3 = -1.0;
        assert!(matches!(
            request.validate(),
            Err(RequestError::NegativeDopingMagnitude { .. })
        ));
    }

    #[test]
    fn mixed_sign_semantics_are_not_inferred() {
        let mut request = fixture();
        request.p_region.donor_density_per_m3 = 1.0e20;
        assert_eq!(
            request.validate(),
            Err(RequestError::AmbiguousV1DopingProfile)
        );
    }

    #[test]
    fn interior_anode_is_rejected() {
        let mut request = fixture();
        request.anode.position_m = 1.0e-9;
        assert_eq!(request.validate(), Err(RequestError::AnodePlacementMismatch));
    }

    #[test]
    fn out_of_envelope_bias_is_rejected() {
        let mut request = fixture();
        request.bias_sweep.stop_volts = 1.2;
        request.bias_sweep.step_volts = 0.1;
        assert_eq!(
            request.validate(),
            Err(RequestError::BiasOutsideV1Envelope)
        );
    }

    #[test]
    fn bias_step_must_land_on_stop() {
        let mut request = fixture();
        request.bias_sweep.step_volts = 0.07;
        assert_eq!(
            request.validate(),
            Err(RequestError::BiasStepDoesNotLandOnStop)
        );
    }

    #[test]
    fn excessive_bias_point_count_is_rejected() {
        let mut request = fixture();
        request.bias_sweep.start_volts = 0.0;
        request.bias_sweep.stop_volts = 1.0;
        request.bias_sweep.step_volts = 0.0005;
        assert_eq!(
            request.validate(),
            Err(RequestError::BiasPointBudgetExceeded(2_001))
        );
    }

    #[test]
    fn region_boundary_drift_is_rejected() {
        let mut request = fixture();
        request.p_region.end_m = 4.9e-7;
        assert_eq!(
            request.validate(),
            Err(RequestError::RegionBoundaryMismatch)
        );
    }

    #[test]
    fn noncanonical_subject_reference_is_rejected() {
        let mut request = fixture();
        request.subject_ref.push(' ');
        assert!(matches!(
            request.validate(),
            Err(RequestError::NonCanonicalIdentifier {
                kind: "subject_ref",
                ..
            })
        ));
    }
}
