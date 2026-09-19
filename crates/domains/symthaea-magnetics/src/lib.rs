// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence contracts for permanent-magnet materials discovery.
//!
//! Intrinsic magnetic properties such as saturation polarization and
//! magnetocrystalline anisotropy are not coercivity. Extrinsic permanent-magnet
//! performance depends on processing, microstructure, interfaces, specimen state,
//! and operating conditions. This crate keeps those evidence planes separate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use symthaea_materials::{
    ConditionedPropertyError, ConditionedPropertyObservation, PropertyArtifactRef,
};
use thiserror::Error;

const PHASE_SUM_TOLERANCE: f64 = 1.0e-6;

/// Whether a magnetic quantity is primarily an intrinsic material descriptor or an
/// extrinsic performance quantity requiring a microstructure/sample context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MagneticEvidencePlane {
    /// Electronic/crystal-scale property that can be evaluated without a bulk magnet microstructure.
    Intrinsic,
    /// Performance quantity whose interpretation requires a microstructure/sample context.
    Extrinsic,
}

/// Canonical magnetic properties used by the rare-earth-free magnet program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MagneticPropertyKind {
    /// Saturation polarization `J_s`, canonical unit tesla.
    SaturationPolarization,
    /// Saturation magnetization `M_s`, canonical unit A/m.
    SaturationMagnetization,
    /// First-order magnetocrystalline anisotropy constant `K1`, canonical unit J/m^3.
    MagnetocrystallineAnisotropyK1,
    /// Curie temperature, canonical unit K.
    CurieTemperature,
    /// Exchange stiffness, canonical unit J/m.
    ExchangeStiffness,
    /// Coercive field, canonical unit A/m.
    CoerciveField,
    /// Remanent flux density, canonical unit T.
    RemanentFluxDensity,
    /// Maximum energy product `(BH)max`, canonical unit J/m^3.
    MaximumEnergyProduct,
    /// Hysteresis-loop energy loss per cycle and volume, canonical unit J/m^3.
    HysteresisLoss,
}

impl MagneticPropertyKind {
    /// Evidence plane for this property.
    pub fn plane(self) -> MagneticEvidencePlane {
        match self {
            Self::SaturationPolarization
            | Self::SaturationMagnetization
            | Self::MagnetocrystallineAnisotropyK1
            | Self::CurieTemperature
            | Self::ExchangeStiffness => MagneticEvidencePlane::Intrinsic,
            Self::CoerciveField
            | Self::RemanentFluxDensity
            | Self::MaximumEnergyProduct
            | Self::HysteresisLoss => MagneticEvidencePlane::Extrinsic,
        }
    }

    /// Canonical MAT-008 property identifier.
    pub fn property_id(self) -> &'static str {
        match self {
            Self::SaturationPolarization => "saturation_polarization",
            Self::SaturationMagnetization => "saturation_magnetization",
            Self::MagnetocrystallineAnisotropyK1 => "magnetocrystalline_anisotropy_k1",
            Self::CurieTemperature => "curie_temperature",
            Self::ExchangeStiffness => "exchange_stiffness",
            Self::CoerciveField => "coercive_field",
            Self::RemanentFluxDensity => "remanent_flux_density",
            Self::MaximumEnergyProduct => "maximum_energy_product",
            Self::HysteresisLoss => "hysteresis_loss",
        }
    }

    /// Canonical unit. Provider/domain adapters must explicitly convert other units.
    pub fn canonical_unit(self) -> &'static str {
        match self {
            Self::SaturationPolarization | Self::RemanentFluxDensity => "T",
            Self::SaturationMagnetization | Self::CoerciveField => "A/m",
            Self::MagnetocrystallineAnisotropyK1
            | Self::MaximumEnergyProduct
            | Self::HysteresisLoss => "J/m3",
            Self::CurieTemperature => "K",
            Self::ExchangeStiffness => "J/m",
        }
    }
}

/// Phase fraction resolved in an experimentally or computationally characterized microstructure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagneticPhaseFraction {
    /// Stable phase identifier.
    pub phase_id: String,
    /// Volume fraction in `[0,1]`.
    pub volume_fraction: f64,
}

/// Provenance-bound microstructure state needed for extrinsic permanent-magnet performance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagneticMicrostructureContext {
    /// Stable microstructure-state identifier.
    pub microstructure_id: String,
    /// MAT-013 physical sample/process-lineage node.
    pub sample_lineage_node_id: String,
    /// Processing-state identifier or exact route/version associated with this state.
    pub process_state_id: String,
    /// Exact characterization/reconstruction artifact for this microstructure.
    pub microstructure_artifact: PropertyArtifactRef,
    /// Resolved phase fractions; when supplied, they must sum to one within tolerance.
    pub phase_fractions: Vec<MagneticPhaseFraction>,
    /// Mean/effective grain size, nm.
    pub mean_grain_size_nm: Option<f64>,
    /// Mean crystallographic/magnetic misorientation, degrees.
    pub mean_misorientation_deg: Option<f64>,
    /// Porosity volume fraction in `[0,1]`.
    pub porosity_fraction: Option<f64>,
    /// Interphase/domain-relevant interface area per volume, m^-1.
    pub interface_density_m_inv: Option<f64>,
}

impl MagneticMicrostructureContext {
    /// Validate physical ranges and lineage/provenance bindings.
    pub fn validate(&self) -> Result<(), MagneticsError> {
        nonempty("microstructure_id", &self.microstructure_id)?;
        nonempty("sample_lineage_node_id", &self.sample_lineage_node_id)?;
        nonempty("process_state_id", &self.process_state_id)?;
        validate_artifact(&self.microstructure_artifact)?;

        let mut phase_ids = HashSet::new();
        let mut phase_sum = 0.0;
        for phase in &self.phase_fractions {
            nonempty("phase_id", &phase.phase_id)?;
            fraction("phase volume_fraction", phase.volume_fraction)?;
            if !phase_ids.insert(phase.phase_id.as_str()) {
                return Err(MagneticsError::DuplicatePhaseId(phase.phase_id.clone()));
            }
            phase_sum += phase.volume_fraction;
        }
        if !self.phase_fractions.is_empty() && (phase_sum - 1.0).abs() > PHASE_SUM_TOLERANCE {
            return Err(MagneticsError::PhaseFractionsDoNotSumToOne { sum: phase_sum });
        }

        if let Some(value) = self.mean_grain_size_nm {
            positive("mean_grain_size_nm", value)?;
        }
        if let Some(value) = self.mean_misorientation_deg {
            finite("mean_misorientation_deg", value)?;
            if !(0.0..=180.0).contains(&value) {
                return Err(MagneticsError::MisorientationOutOfRange(value));
            }
        }
        if let Some(value) = self.porosity_fraction {
            fraction("porosity_fraction", value)?;
        }
        if let Some(value) = self.interface_density_m_inv {
            nonnegative("interface_density_m_inv", value)?;
        }
        Ok(())
    }

    /// Deterministic comparison identity for this exact characterized microstructure state.
    pub fn canonical_identity(&self) -> Result<String, MagneticsError> {
        self.validate()?;
        let mut phases = self.phase_fractions.clone();
        phases.sort_by(|a, b| a.phase_id.cmp(&b.phase_id));
        let phases = phases
            .iter()
            .map(|phase| format!("{}:{}", token(&phase.phase_id), float_key(phase.volume_fraction)))
            .collect::<Vec<_>>()
            .join(",");
        Ok(format!(
            "magnetic-microstructure:v1|id={}|sample={}|process={}|artifact={}|phases=[{}]|grain_nm={}|misorientation_deg={}|porosity={}|interface_m_inv={}",
            token(&self.microstructure_id),
            token(&self.sample_lineage_node_id),
            token(&self.process_state_id),
            self.microstructure_artifact.artifact_sha256.to_ascii_lowercase(),
            phases,
            option_float(self.mean_grain_size_nm),
            option_float(self.mean_misorientation_deg),
            option_float(self.porosity_fraction),
            option_float(self.interface_density_m_inv),
        ))
    }
}

/// One magnetic-property evidence item layered over MAT-008 conditioned evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagneticPropertyEvidence {
    /// Semantic magnetic property.
    pub kind: MagneticPropertyKind,
    /// Exact conditioned property observation.
    pub observation: ConditionedPropertyObservation,
    /// Optional intrinsic context; mandatory for extrinsic performance quantities.
    pub microstructure: Option<MagneticMicrostructureContext>,
}

impl MagneticPropertyEvidence {
    /// Validate property identity/unit and intrinsic-vs-extrinsic evidence boundaries.
    pub fn validate(&self) -> Result<(), MagneticsError> {
        self.observation.validate()?;
        if self.observation.property_id != self.kind.property_id() {
            return Err(MagneticsError::PropertyKindMismatch {
                expected: self.kind.property_id().to_string(),
                actual: self.observation.property_id.clone(),
            });
        }
        if self.observation.unit != self.kind.canonical_unit() {
            return Err(MagneticsError::NonCanonicalUnit {
                property_id: self.kind.property_id().to_string(),
                expected: self.kind.canonical_unit().to_string(),
                actual: self.observation.unit.clone(),
            });
        }

        if let Some(context) = &self.microstructure {
            context.validate()?;
        }
        if self.kind.plane() == MagneticEvidencePlane::Extrinsic {
            if self.microstructure.is_none() {
                return Err(MagneticsError::MicrostructureRequired {
                    property_id: self.kind.property_id().to_string(),
                });
            }
            if self.observation.conditions.temperature_k.is_none() {
                return Err(MagneticsError::ExtrinsicTemperatureRequired {
                    property_id: self.kind.property_id().to_string(),
                });
            }
        }
        Ok(())
    }

    /// Conservative direct-comparison key including microstructure when present.
    pub fn direct_comparison_key(&self) -> Result<String, MagneticsError> {
        self.validate()?;
        let microstructure = match &self.microstructure {
            Some(context) => context.canonical_identity()?,
            None => "none".to_string(),
        };
        Ok(format!(
            "magnetic-evidence:v1|plane={:?}|property={}|observation={}|microstructure={}",
            self.kind.plane(),
            self.kind.property_id(),
            token(&self.observation.direct_comparison_key()?),
            token(&microstructure),
        ))
    }
}

fn validate_artifact(value: &PropertyArtifactRef) -> Result<(), MagneticsError> {
    nonempty("artifact source_id", &value.source_id)?;
    if value.artifact_sha256.len() != 64
        || !value.artifact_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(MagneticsError::InvalidSha256);
    }
    Ok(())
}

fn nonempty(field: &'static str, value: &str) -> Result<(), MagneticsError> {
    if value.trim().is_empty() {
        Err(MagneticsError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), MagneticsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MagneticsError::NonFiniteValue { field, value })
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), MagneticsError> {
    finite(field, value)?;
    if value < 0.0 {
        Err(MagneticsError::NegativeValue { field, value })
    } else {
        Ok(())
    }
}

fn positive(field: &'static str, value: f64) -> Result<(), MagneticsError> {
    finite(field, value)?;
    if value <= 0.0 {
        Err(MagneticsError::NonPositiveValue { field, value })
    } else {
        Ok(())
    }
}

fn fraction(field: &'static str, value: f64) -> Result<(), MagneticsError> {
    finite(field, value)?;
    if !(0.0..=1.0).contains(&value) {
        Err(MagneticsError::FractionOutOfRange { field, value })
    } else {
        Ok(())
    }
}

fn float_key(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn option_float(value: Option<f64>) -> String {
    value.map(float_key).unwrap_or_else(|| "unknown".to_string())
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Magnetics evidence validation failure.
#[derive(Debug, Error)]
pub enum MagneticsError {
    /// Underlying MAT-008 observation was invalid.
    #[error("invalid conditioned property observation: {0:?}")]
    ConditionedProperty(ConditionedPropertyError),
    /// Required text was empty.
    #[error("required magnetics field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 binding malformed.
    #[error("invalid SHA-256")]
    InvalidSha256,
    /// Numeric value not finite.
    #[error("non-finite value in {field}: {value}")]
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Negative value not permitted.
    #[error("negative value in {field}: {value}")]
    NegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Zero/negative value not permitted.
    #[error("non-positive value in {field}: {value}")]
    NonPositiveValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Fraction outside [0,1].
    #[error("fraction outside [0,1] in {field}: {value}")]
    FractionOutOfRange {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Duplicate phase identifier.
    #[error("duplicate magnetic phase ID: {0}")]
    DuplicatePhaseId(String),
    /// Resolved phase fractions did not sum to unity.
    #[error("magnetic phase fractions must sum to one; got {sum}")]
    PhaseFractionsDoNotSumToOne {
        /// Observed sum.
        sum: f64,
    },
    /// Misorientation must be between 0 and 180 degrees.
    #[error("mean misorientation outside [0,180] degrees: {0}")]
    MisorientationOutOfRange(f64),
    /// Magnetic property kind and MAT-008 property ID disagree.
    #[error("magnetic property kind mismatch: expected {expected}, got {actual}")]
    PropertyKindMismatch {
        /// Expected property ID.
        expected: String,
        /// Actual property ID.
        actual: String,
    },
    /// Observation did not use the canonical unit for this magnetics contract.
    #[error("non-canonical unit for {property_id}: expected {expected}, got {actual}")]
    NonCanonicalUnit {
        /// Property ID.
        property_id: String,
        /// Canonical unit.
        expected: String,
        /// Actual unit.
        actual: String,
    },
    /// Extrinsic performance cannot be interpreted without microstructure/sample lineage.
    #[error("microstructure is required for extrinsic magnetic property: {property_id}")]
    MicrostructureRequired {
        /// Property ID.
        property_id: String,
    },
    /// Extrinsic performance observation did not specify temperature.
    #[error("temperature is required for extrinsic magnetic property: {property_id}")]
    ExtrinsicTemperatureRequired {
        /// Property ID.
        property_id: String,
    },
}

impl From<ConditionedPropertyError> for MagneticsError {
    fn from(value: ConditionedPropertyError) -> Self {
        Self::ConditionedProperty(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials::{
        PropertyConditions, PropertyObservationMethod, PropertyUncertainty,
    };

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn artifact(source: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn dft_observation(property_id: &str, value: f64, unit: &str) -> ConditionedPropertyObservation {
        ConditionedPropertyObservation {
            subject_identity: "material-subject:v1|Fe5Co18Zr6-fixture".to_string(),
            property_id: property_id.to_string(),
            value,
            unit: unit.to_string(),
            uncertainty: PropertyUncertainty::Unknown,
            conditions: PropertyConditions {
                temperature_k: Some(0.0),
                ..Default::default()
            },
            method: PropertyObservationMethod::Calculation {
                method_id: "spin-orbit-dft-pbe".to_string(),
                code_id: "dft-fixture".to_string(),
                input_sha256: A64.to_string(),
                output_sha256: B64.to_string(),
            },
            artifact: artifact("dft-output", B64),
        }
    }

    fn microstructure(id: &str, grain_nm: f64) -> MagneticMicrostructureContext {
        MagneticMicrostructureContext {
            microstructure_id: id.to_string(),
            sample_lineage_node_id: format!("sample-{id}"),
            process_state_id: "anneal-v1".to_string(),
            microstructure_artifact: artifact("microstructure", A64),
            phase_fractions: vec![MagneticPhaseFraction {
                phase_id: "target-phase".to_string(),
                volume_fraction: 1.0,
            }],
            mean_grain_size_nm: Some(grain_nm),
            mean_misorientation_deg: Some(5.0),
            porosity_fraction: Some(0.01),
            interface_density_m_inv: Some(1.0e7),
        }
    }

    fn coercivity_observation() -> ConditionedPropertyObservation {
        ConditionedPropertyObservation {
            subject_identity: "material-subject:v1|same-composition".to_string(),
            property_id: "coercive_field".to_string(),
            value: 450_000.0,
            unit: "A/m".to_string(),
            uncertainty: PropertyUncertainty::Unknown,
            conditions: PropertyConditions {
                temperature_k: Some(300.0),
                geometry_id: Some("vsm-coupon-v1".to_string()),
                ..Default::default()
            },
            method: PropertyObservationMethod::Experiment {
                method_id: "hysteresis-loop-v1".to_string(),
                instrument_id: Some("vsm-fixture".to_string()),
                calibration: Some(artifact("vsm-calibration", A64)),
            },
            artifact: artifact("hysteresis-raw", B64),
        }
    }

    #[test]
    fn intrinsic_k1_accepts_zero_kelvin_without_microstructure() {
        let evidence = MagneticPropertyEvidence {
            kind: MagneticPropertyKind::MagnetocrystallineAnisotropyK1,
            observation: dft_observation("magnetocrystalline_anisotropy_k1", 1.1e6, "J/m3"),
            microstructure: None,
        };
        evidence.validate().unwrap();
        assert_eq!(evidence.kind.plane(), MagneticEvidencePlane::Intrinsic);
    }

    #[test]
    fn high_k1_cannot_be_relabelled_as_coercivity() {
        let evidence = MagneticPropertyEvidence {
            kind: MagneticPropertyKind::CoerciveField,
            observation: dft_observation("magnetocrystalline_anisotropy_k1", 1.1e6, "J/m3"),
            microstructure: Some(microstructure("a", 100.0)),
        };
        assert!(matches!(
            evidence.validate(),
            Err(MagneticsError::PropertyKindMismatch { .. })
        ));
    }

    #[test]
    fn coercivity_requires_bound_microstructure_lineage() {
        let evidence = MagneticPropertyEvidence {
            kind: MagneticPropertyKind::CoerciveField,
            observation: coercivity_observation(),
            microstructure: None,
        };
        assert!(matches!(
            evidence.validate(),
            Err(MagneticsError::MicrostructureRequired { .. })
        ));
    }

    #[test]
    fn extrinsic_performance_requires_temperature() {
        let mut observation = coercivity_observation();
        observation.conditions.temperature_k = None;
        let evidence = MagneticPropertyEvidence {
            kind: MagneticPropertyKind::CoerciveField,
            observation,
            microstructure: Some(microstructure("a", 100.0)),
        };
        assert!(matches!(
            evidence.validate(),
            Err(MagneticsError::ExtrinsicTemperatureRequired { .. })
        ));
    }

    #[test]
    fn same_composition_different_microstructure_is_not_same_performance_context() {
        let left = MagneticPropertyEvidence {
            kind: MagneticPropertyKind::CoerciveField,
            observation: coercivity_observation(),
            microstructure: Some(microstructure("fine-grain", 80.0)),
        };
        let right = MagneticPropertyEvidence {
            kind: MagneticPropertyKind::CoerciveField,
            observation: coercivity_observation(),
            microstructure: Some(microstructure("coarse-grain", 800.0)),
        };
        assert_ne!(
            left.direct_comparison_key().unwrap(),
            right.direct_comparison_key().unwrap()
        );
    }

    #[test]
    fn resolved_phase_fractions_must_sum_to_one() {
        let mut context = microstructure("bad-phases", 100.0);
        context.phase_fractions = vec![
            MagneticPhaseFraction {
                phase_id: "a".to_string(),
                volume_fraction: 0.7,
            },
            MagneticPhaseFraction {
                phase_id: "b".to_string(),
                volume_fraction: 0.2,
            },
        ];
        assert!(matches!(
            context.validate(),
            Err(MagneticsError::PhaseFractionsDoNotSumToOne { .. })
        ));
    }
}
