// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Condition-bearing material-property evidence.
//!
//! A property value without its state and measurement/calculation conditions is
//! incomplete evidence. Thermal conductivity, coercivity, ionic conductivity,
//! strength, catalytic activity, permeability, and many other useful quantities
//! depend on temperature, pressure, atmosphere, field, frequency, specimen state,
//! geometry, age, or contact conditions.
//!
//! This module stores those conditions explicitly and distinguishes experiments,
//! calculations, imported reports, and model predictions. It does not itself
//! advance MAT-001 evidence authority.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const SHA256_HEX_LEN: usize = 64;

/// Source artifact binding for one property observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyArtifactRef {
    /// Stable source identifier.
    pub source_id: String,
    /// SHA-256 digest of the exact source, raw file, captured response, or result artifact.
    pub artifact_sha256: String,
}

/// Additional condition not covered by the common typed fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyConditionTag {
    /// Stable condition key such as `ph`, `grain_size_nm`, or `bond_line_um`.
    pub key: String,
    /// Numeric value.
    pub value: f64,
    /// Explicit unit or dimensionless marker.
    pub unit: String,
}

/// Environmental, loading, temporal, and geometric state for a property observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PropertyConditions {
    /// Temperature in kelvin.
    pub temperature_k: Option<f64>,
    /// Pressure in pascals.
    pub pressure_pa: Option<f64>,
    /// Named atmosphere/environment, for example `vacuum`, `argon`, or `air`.
    pub atmosphere: Option<String>,
    /// Applied magnetic field in tesla.
    pub magnetic_field_t: Option<f64>,
    /// Applied electric field in V/m.
    pub electric_field_v_m: Option<f64>,
    /// Measurement/excitation frequency in hertz.
    pub frequency_hz: Option<f64>,
    /// Mechanical strain rate in s^-1.
    pub strain_rate_s_inv: Option<f64>,
    /// Relative humidity as a fraction in [0,1].
    pub relative_humidity: Option<f64>,
    /// Material/sample age or curing time in seconds.
    pub age_seconds: Option<f64>,
    /// Contact pressure in pascals, especially relevant to interface properties.
    pub contact_pressure_pa: Option<f64>,
    /// Orientation/texture/loading direction identifier.
    pub orientation_id: Option<String>,
    /// Specimen/device geometry identifier or bound protocol key.
    pub geometry_id: Option<String>,
    /// Extra numeric conditions, keyed explicitly rather than hidden in free text.
    pub extra: Vec<PropertyConditionTag>,
}

impl PropertyConditions {
    /// Validate all supplied state variables and extra tags.
    pub fn validate(&self) -> Result<(), ConditionedPropertyError> {
        if let Some(value) = self.temperature_k {
            validate_positive_finite("temperature_k", value)?;
        }
        if let Some(value) = self.pressure_pa {
            validate_nonnegative_finite("pressure_pa", value)?;
        }
        if let Some(value) = self.magnetic_field_t {
            validate_finite("magnetic_field_t", value)?;
        }
        if let Some(value) = self.electric_field_v_m {
            validate_finite("electric_field_v_m", value)?;
        }
        if let Some(value) = self.frequency_hz {
            validate_positive_finite("frequency_hz", value)?;
        }
        if let Some(value) = self.strain_rate_s_inv {
            validate_nonnegative_finite("strain_rate_s_inv", value)?;
        }
        if let Some(value) = self.relative_humidity {
            validate_finite("relative_humidity", value)?;
            if !(0.0..=1.0).contains(&value) {
                return Err(ConditionedPropertyError::FractionOutOfRange {
                    field: "relative_humidity",
                    value,
                });
            }
        }
        if let Some(value) = self.age_seconds {
            validate_nonnegative_finite("age_seconds", value)?;
        }
        if let Some(value) = self.contact_pressure_pa {
            validate_nonnegative_finite("contact_pressure_pa", value)?;
        }
        if let Some(value) = &self.atmosphere {
            validate_nonempty("atmosphere", value)?;
        }
        if let Some(value) = &self.orientation_id {
            validate_nonempty("orientation_id", value)?;
        }
        if let Some(value) = &self.geometry_id {
            validate_nonempty("geometry_id", value)?;
        }

        let mut keys = HashSet::new();
        for tag in &self.extra {
            validate_nonempty("condition key", &tag.key)?;
            validate_nonempty("condition unit", &tag.unit)?;
            validate_finite("condition value", tag.value)?;
            if !keys.insert(tag.key.as_str()) {
                return Err(ConditionedPropertyError::DuplicateConditionKey(
                    tag.key.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Exact deterministic condition signature used for conservative direct comparison.
    ///
    /// Numeric values are encoded by IEEE-754 bit pattern so no decimal formatting
    /// or tolerance is silently introduced. A domain adapter may later define an
    /// explicit transformation or tolerance when scientifically justified.
    pub fn canonical_signature(&self) -> Result<String, ConditionedPropertyError> {
        self.validate()?;
        let mut extra = self.extra.clone();
        extra.sort_by(|a, b| a.key.cmp(&b.key));
        let extra = extra
            .iter()
            .map(|tag| {
                format!(
                    "{}={}:{}",
                    encode_token(&tag.key),
                    float_key(tag.value),
                    encode_token(&tag.unit)
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        Ok(format!(
            "T={}|P={}|atm={}|B={}|E={}|f={}|strain_rate={}|rh={}|age={}|contactP={}|orient={}|geom={}|extra=[{}]",
            optional_float_key(self.temperature_k),
            optional_float_key(self.pressure_pa),
            optional_string_key(self.atmosphere.as_deref()),
            optional_float_key(self.magnetic_field_t),
            optional_float_key(self.electric_field_v_m),
            optional_float_key(self.frequency_hz),
            optional_float_key(self.strain_rate_s_inv),
            optional_float_key(self.relative_humidity),
            optional_float_key(self.age_seconds),
            optional_float_key(self.contact_pressure_pa),
            optional_string_key(self.orientation_id.as_deref()),
            optional_string_key(self.geometry_id.as_deref()),
            extra
        ))
    }
}

/// Explicit uncertainty representation for a property value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyUncertainty {
    /// Source provided no quantitative uncertainty.
    Unknown,
    /// One-standard-uncertainty estimate in the same unit as the property value.
    Standard {
        /// Non-negative standard uncertainty.
        sigma: f64,
    },
    /// Explicit interval in the same unit as the property value.
    Interval {
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
        /// Optional confidence/coverage fraction in (0,1].
        confidence_fraction: Option<f64>,
    },
}

impl PropertyUncertainty {
    fn validate(&self, value: f64) -> Result<(), ConditionedPropertyError> {
        match self {
            Self::Unknown => Ok(()),
            Self::Standard { sigma } => validate_nonnegative_finite("sigma", *sigma),
            Self::Interval {
                lower,
                upper,
                confidence_fraction,
            } => {
                validate_finite("uncertainty lower", *lower)?;
                validate_finite("uncertainty upper", *upper)?;
                if lower > upper {
                    return Err(ConditionedPropertyError::InvalidUncertaintyInterval);
                }
                if value < *lower || value > *upper {
                    return Err(ConditionedPropertyError::ValueOutsideUncertaintyInterval);
                }
                if let Some(confidence) = confidence_fraction {
                    validate_finite("confidence_fraction", *confidence)?;
                    if *confidence <= 0.0 || *confidence > 1.0 {
                        return Err(ConditionedPropertyError::FractionOutOfRange {
                            field: "confidence_fraction",
                            value: *confidence,
                        });
                    }
                }
                Ok(())
            }
        }
    }
}

/// How a property value was obtained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyObservationMethod {
    /// Physical measurement on a material specimen/device.
    Experiment {
        /// Named test/measurement protocol.
        method_id: String,
        /// Optional instrument identifier.
        instrument_id: Option<String>,
        /// Optional bound calibration artifact.
        calibration: Option<PropertyArtifactRef>,
    },
    /// First-principles, atomistic, continuum, or other explicit calculation.
    Calculation {
        /// Method identifier such as `DFT-PBE`, `DFPT`, `CALPHAD`, or `FEM`.
        method_id: String,
        /// Exact code/environment identity.
        code_id: String,
        /// SHA-256 of calculation inputs.
        input_sha256: String,
        /// SHA-256 of calculation outputs.
        output_sha256: String,
    },
    /// Value reported in a publication or other literature source.
    LiteratureReported {
        /// Publication/report identifier.
        publication_id: String,
        /// Table, figure, page, or textual locator.
        locator: String,
    },
    /// Value imported from an external database/provider.
    DatabaseImported {
        /// Provider identifier.
        provider_id: String,
        /// Provider record/material identifier.
        record_id: String,
    },
    /// Prediction from a statistical/ML/surrogate model.
    ModelPrediction {
        /// Stable model identity.
        model_id: String,
        /// SHA-256 of exact model weights/artifact.
        model_sha256: String,
        /// Optional applicability-domain or calibration-set identifier.
        applicability_domain_id: Option<String>,
    },
}

impl PropertyObservationMethod {
    fn validate(&self) -> Result<(), ConditionedPropertyError> {
        match self {
            Self::Experiment {
                method_id,
                instrument_id,
                calibration,
            } => {
                validate_nonempty("method_id", method_id)?;
                if let Some(instrument) = instrument_id {
                    validate_nonempty("instrument_id", instrument)?;
                }
                if let Some(calibration) = calibration {
                    validate_artifact(calibration)?;
                }
                Ok(())
            }
            Self::Calculation {
                method_id,
                code_id,
                input_sha256,
                output_sha256,
            } => {
                validate_nonempty("method_id", method_id)?;
                validate_nonempty("code_id", code_id)?;
                validate_sha256(input_sha256)?;
                validate_sha256(output_sha256)
            }
            Self::LiteratureReported {
                publication_id,
                locator,
            } => {
                validate_nonempty("publication_id", publication_id)?;
                validate_nonempty("locator", locator)
            }
            Self::DatabaseImported {
                provider_id,
                record_id,
            } => {
                validate_nonempty("provider_id", provider_id)?;
                validate_nonempty("record_id", record_id)
            }
            Self::ModelPrediction {
                model_id,
                model_sha256,
                applicability_domain_id,
            } => {
                validate_nonempty("model_id", model_id)?;
                validate_sha256(model_sha256)?;
                if let Some(domain) = applicability_domain_id {
                    validate_nonempty("applicability_domain_id", domain)?;
                }
                Ok(())
            }
        }
    }

    /// Broad descriptive evidence class. This is not MAT-001 authority.
    pub fn evidence_class(&self) -> PropertyEvidenceClass {
        match self {
            Self::Experiment { .. } => PropertyEvidenceClass::Experimental,
            Self::Calculation { .. } => PropertyEvidenceClass::Calculated,
            Self::LiteratureReported { .. } => PropertyEvidenceClass::Reported,
            Self::DatabaseImported { .. } => PropertyEvidenceClass::Imported,
            Self::ModelPrediction { .. } => PropertyEvidenceClass::Predicted,
        }
    }
}

/// Descriptive origin class for a property value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyEvidenceClass {
    /// Physical experiment.
    Experimental,
    /// Explicit physics/chemistry calculation.
    Calculated,
    /// Literature-reported value.
    Reported,
    /// External database value.
    Imported,
    /// ML/statistical/surrogate prediction.
    Predicted,
}

/// One source-bound property observation for one exact MAT-007 subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionedPropertyObservation {
    /// Exact canonical MAT-007 subject identity string.
    pub subject_identity: String,
    /// Stable property identifier, for example `thermal_conductivity`, `coercivity`,
    /// `ionic_conductivity`, `compressive_strength_28d`, or `critical_temperature`.
    pub property_id: String,
    /// Numeric property value.
    pub value: f64,
    /// Explicit property unit. Unit conversion is never implicit here.
    pub unit: String,
    /// Quantitative uncertainty when available.
    pub uncertainty: PropertyUncertainty,
    /// State/loading/environment conditions.
    pub conditions: PropertyConditions,
    /// Method by which the value was obtained.
    pub method: PropertyObservationMethod,
    /// Exact source/result artifact binding.
    pub artifact: PropertyArtifactRef,
}

impl ConditionedPropertyObservation {
    /// Validate the observation without assigning scientific authority.
    pub fn validate(&self) -> Result<(), ConditionedPropertyError> {
        validate_nonempty("subject_identity", &self.subject_identity)?;
        validate_nonempty("property_id", &self.property_id)?;
        validate_nonempty("unit", &self.unit)?;
        validate_finite("property value", self.value)?;
        self.uncertainty.validate(self.value)?;
        self.conditions.validate()?;
        self.method.validate()?;
        validate_artifact(&self.artifact)
    }

    /// Conservative direct-comparison key.
    ///
    /// Values are directly comparable by this module only when subject, property,
    /// unit, and exact conditions all match. Different methods may still be compared
    /// as independent observations under identical conditions; callers remain
    /// responsible for interpreting systematic method differences.
    pub fn direct_comparison_key(&self) -> Result<String, ConditionedPropertyError> {
        self.validate()?;
        Ok(format!(
            "subject={}|property={}|unit={}|conditions={}",
            encode_token(&self.subject_identity),
            encode_token(&self.property_id),
            encode_token(&self.unit),
            self.conditions.canonical_signature()?
        ))
    }

    /// Whether two values are directly comparable without an explicit normalization
    /// or condition transformation supplied by a higher-level domain adapter.
    pub fn is_directly_comparable_with(
        &self,
        other: &Self,
    ) -> Result<bool, ConditionedPropertyError> {
        Ok(self.direct_comparison_key()? == other.direct_comparison_key()?)
    }
}

fn validate_artifact(artifact: &PropertyArtifactRef) -> Result<(), ConditionedPropertyError> {
    validate_nonempty("source_id", &artifact.source_id)?;
    validate_sha256(&artifact.artifact_sha256)
}

fn validate_sha256(value: &str) -> Result<(), ConditionedPropertyError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(ConditionedPropertyError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn validate_nonempty(field: &'static str, value: &str) -> Result<(), ConditionedPropertyError> {
    if value.trim().is_empty() {
        Err(ConditionedPropertyError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), ConditionedPropertyError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConditionedPropertyError::NonFiniteValue { field, value })
    }
}

fn validate_nonnegative_finite(
    field: &'static str,
    value: f64,
) -> Result<(), ConditionedPropertyError> {
    validate_finite(field, value)?;
    if value < 0.0 {
        Err(ConditionedPropertyError::NegativeValue { field, value })
    } else {
        Ok(())
    }
}

fn validate_positive_finite(
    field: &'static str,
    value: f64,
) -> Result<(), ConditionedPropertyError> {
    validate_finite(field, value)?;
    if value <= 0.0 {
        Err(ConditionedPropertyError::NonPositiveValue { field, value })
    } else {
        Ok(())
    }
}

fn optional_float_key(value: Option<f64>) -> String {
    value.map(float_key).unwrap_or_else(|| "unknown".to_string())
}

fn optional_string_key(value: Option<&str>) -> String {
    value
        .map(encode_token)
        .unwrap_or_else(|| "unknown".to_string())
}

fn float_key(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn encode_token(value: &str) -> String {
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

/// Conditioned-property validation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionedPropertyError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// Numeric value was NaN or infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Numeric value was negative where non-negative was required.
    NegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Numeric value was zero/negative where strictly positive was required.
    NonPositiveValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Fraction-like value was outside its allowed range.
    FractionOutOfRange {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Additional condition key was repeated.
    DuplicateConditionKey(String),
    /// Uncertainty interval had lower > upper.
    InvalidUncertaintyInterval,
    /// Reported value fell outside its declared uncertainty interval.
    ValueOutsideUncertaintyInterval,
    /// Artifact digest was not 64 hexadecimal characters.
    InvalidSha256,
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn artifact(hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: "fixture-source".to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn observation() -> ConditionedPropertyObservation {
        ConditionedPropertyObservation {
            subject_identity: "material-subject:v1|fixture".to_string(),
            property_id: "thermal_conductivity".to_string(),
            value: 123.4,
            unit: "W/(m*K)".to_string(),
            uncertainty: PropertyUncertainty::Standard { sigma: 1.2 },
            conditions: PropertyConditions {
                temperature_k: Some(300.0),
                pressure_pa: Some(101_325.0),
                atmosphere: Some("argon".to_string()),
                contact_pressure_pa: Some(1_000_000.0),
                geometry_id: Some("tim-astm-fixture-v1".to_string()),
                ..PropertyConditions::default()
            },
            method: PropertyObservationMethod::Experiment {
                method_id: "steady-state-thermal-v1".to_string(),
                instrument_id: Some("rig-7".to_string()),
                calibration: Some(artifact(B64)),
            },
            artifact: artifact(A64),
        }
    }

    #[test]
    fn valid_observation_accepts_explicit_conditions() {
        observation().validate().unwrap();
    }

    #[test]
    fn temperature_change_prevents_silent_direct_comparison() {
        let a = observation();
        let mut b = observation();
        b.conditions.temperature_k = Some(500.0);
        assert!(!a.is_directly_comparable_with(&b).unwrap());
    }

    #[test]
    fn contact_pressure_change_prevents_silent_tim_comparison() {
        let a = observation();
        let mut b = observation();
        b.conditions.contact_pressure_pa = Some(2_000_000.0);
        assert!(!a.is_directly_comparable_with(&b).unwrap());
    }

    #[test]
    fn method_can_differ_under_identical_conditions() {
        let a = observation();
        let mut b = observation();
        b.method = PropertyObservationMethod::Calculation {
            method_id: "fem-steady-state-v1".to_string(),
            code_id: "solver-build-42".to_string(),
            input_sha256: A64.to_string(),
            output_sha256: B64.to_string(),
        };
        b.artifact = artifact(B64);
        assert!(a.is_directly_comparable_with(&b).unwrap());
        assert_ne!(a.method.evidence_class(), b.method.evidence_class());
    }

    #[test]
    fn unknown_temperature_is_not_same_as_known_temperature() {
        let a = observation();
        let mut b = observation();
        b.conditions.temperature_k = None;
        assert!(!a.is_directly_comparable_with(&b).unwrap());
    }

    #[test]
    fn unit_change_requires_explicit_conversion() {
        let a = observation();
        let mut b = observation();
        b.unit = "mW/(mm*K)".to_string();
        assert!(!a.is_directly_comparable_with(&b).unwrap());
    }

    #[test]
    fn uncertainty_interval_must_contain_value() {
        let mut invalid = observation();
        invalid.uncertainty = PropertyUncertainty::Interval {
            lower: 0.0,
            upper: 100.0,
            confidence_fraction: Some(0.95),
        };
        assert_eq!(
            invalid.validate(),
            Err(ConditionedPropertyError::ValueOutsideUncertaintyInterval)
        );
    }

    #[test]
    fn extra_condition_order_does_not_change_signature() {
        let mut a = observation();
        a.conditions.extra = vec![
            PropertyConditionTag {
                key: "grain_size_nm".to_string(),
                value: 300.0,
                unit: "nm".to_string(),
            },
            PropertyConditionTag {
                key: "porosity".to_string(),
                value: 0.02,
                unit: "fraction".to_string(),
            },
        ];
        let mut b = a.clone();
        b.conditions.extra.reverse();
        assert_eq!(
            a.conditions.canonical_signature().unwrap(),
            b.conditions.canonical_signature().unwrap()
        );
    }

    #[test]
    fn model_prediction_remains_descriptively_predicted() {
        let mut predicted = observation();
        predicted.method = PropertyObservationMethod::ModelPrediction {
            model_id: "surrogate-v3".to_string(),
            model_sha256: A64.to_string(),
            applicability_domain_id: Some("calibration-domain-17".to_string()),
        };
        assert_eq!(
            predicted.method.evidence_class(),
            PropertyEvidenceClass::Predicted
        );
        predicted.validate().unwrap();
    }

    #[test]
    fn relative_humidity_is_bounded() {
        let mut invalid = observation();
        invalid.conditions.relative_humidity = Some(1.2);
        assert!(matches!(
            invalid.validate(),
            Err(ConditionedPropertyError::FractionOutOfRange {
                field: "relative_humidity",
                ..
            })
        ));
    }

    #[test]
    fn source_digest_is_required() {
        let mut invalid = observation();
        invalid.artifact.artifact_sha256 = "not-a-digest".to_string();
        assert_eq!(
            invalid.validate(),
            Err(ConditionedPropertyError::InvalidSha256)
        );
    }
}
