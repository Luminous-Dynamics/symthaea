// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound longitudinal observations for mycelial morphogenesis.
//!
//! Measurements are deliberately typed so optical proxies, cord morphology,
//! directional extension and primordium expansion cannot be silently conflated
//! with biomass, hydraulic conductance, nutrient delivery or causal effects.
//! A single observation represents one measurement in one region at one time.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_acquisition_domain::{
    MycelialAcquisitionDomainManifest, MycelialAcquisitionUnitKey,
    MycelialExtractionCoverageReceipt,
};
use crate::mycelial_observation_coverage::MycelialObservationCoverageLedger;

pub const MYCELIAL_MORPHOGENESIS_OBSERVATION_SCHEMA_VERSION: u16 = 1;
const MAX_MORPHOGENESIS_IDENTIFIER_BYTES: usize = 128;
const MAX_MORPHOGENESIS_UNIT_BYTES: usize = 48;
const MAX_MORPHOGENESIS_METHOD_BYTES: usize = 128;
const MAX_MORPHOGENESIS_STRING_BYTES: usize = 8 * 1024 * 1024;
const DIGEST_HEX_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialMorphogenesisRegionKind {
    SubstratePatch,
    CordSegment,
    HyphalFront,
    Primordium,
    FruitingBody,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialDevelopmentStage {
    VegetativeColonization,
    CasingColonization,
    PrimordiumInitiation,
    ExponentialFruiting,
    Harvest,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialMorphogenesisMissingStatus {
    NotAcquired,
    UnavailableOrCorrupted,
    Unmeasurable,
    OccludedOrOutOfFocus,
    BelowDetection,
    RightCensored,
    LeftCensored,
    IntervalCensored,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MycelialMorphogenesisMeasurement {
    /// Explicit missing/censoring states; never substitute a numeric zero.
    MeasurementUnavailable {
        status: MycelialMorphogenesisMissingStatus,
        reason: String,
    },
    /// Optical intensity is a colonization proxy, not a direct biomass measure.
    ColonizationDensityProxy {
        mean_intensity: f64,
        scale_min: f64,
        scale_max: f64,
        intensity_scale_id: String,
        analysis_method_id: String,
    },
    /// Fraction of the declared substrate/ROI area occupied by segmented cords.
    CordAreaFraction {
        fraction: f64,
        analysis_method_id: String,
    },
    /// Quantiles of measured cord widths; unit and calibration remain explicit.
    CordWidthDistribution {
        sample_count: u32,
        q10: f64,
        median: f64,
        q90: f64,
        unit: String,
        analysis_method_id: String,
    },
    /// Observed displacement vector into the declared coordinate frame.
    DirectionalExtension {
        dx: f64,
        dy: f64,
        dz: f64,
        unit: String,
        analysis_method_id: String,
    },
    /// Area of a primordium or fruiting body; not a direct measure of sink strength.
    PrimordiumArea {
        area: f64,
        unit: String,
        analysis_method_id: String,
    },
    /// Presence/absence from an explicitly named segmentation or annotation method.
    CordPresence {
        present: bool,
        analysis_method_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialMorphogenesisObservation {
    pub schema_version: u16,
    pub observation_id: String,
    /// Digest of the exact source observation/crop in the raw acquisition.
    pub source_observation_digest: String,
    /// Upstream evidence-chain identities.
    pub source_artifact_id: String,
    pub source_artifact_digest: String,
    pub acquisition_domain_manifest_digest: String,
    pub extraction_coverage_receipt_digest: String,
    pub observation_coverage_ledger_digest: String,
    /// Biological unit, acquisition and region identities used for leakage-resistant splits.
    pub biological_replicate_id: String,
    pub cultivation_unit_id: String,
    pub acquisition_session_id: String,
    pub substrate_region_id: String,
    pub region_kind: MycelialMorphogenesisRegionKind,
    pub source_unit_key: MycelialAcquisitionUnitKey,
    /// Monotonic or acquisition-relative timestamp; its declared timebase is mandatory.
    pub timestamp_ns: u64,
    pub timebase_id: String,
    pub development_stage: MycelialDevelopmentStage,
    pub coordinate_frame_id: String,
    pub spatial_transform_digest: String,
    pub calibration_digest: String,
    pub measurement: MycelialMorphogenesisMeasurement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialMorphogenesisObservationError {
    InvalidSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidMeasurement(&'static str),
    InvalidMissingReason,
    EvidenceChainMismatch(&'static str),
    ObservationNotInLedger,
    SourceUnitMismatch,
    ResourceLimitExceeded,
}

impl fmt::Display for MycelialMorphogenesisObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported Mycelial morphogenesis observation schema version {version}"
                )
            }
            Self::InvalidIdentifier(field) => {
                write!(f, "invalid or unbounded morphogenesis identifier: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid morphogenesis evidence digest: {field}")
            }
            Self::InvalidMeasurement(field) => {
                write!(f, "invalid typed morphogenesis measurement: {field}")
            }
            Self::InvalidMissingReason => {
                write!(
                    f,
                    "missing or censored morphogenesis observation requires a bounded reason"
                )
            }
            Self::EvidenceChainMismatch(field) => {
                write!(
                    f,
                    "morphogenesis observation disagrees with evidence chain: {field}"
                )
            }
            Self::ObservationNotInLedger => {
                write!(
                    f,
                    "morphogenesis source observation is absent from the validated observation ledger"
                )
            }
            Self::SourceUnitMismatch => {
                write!(
                    f,
                    "morphogenesis source-unit identity differs from observation ledger"
                )
            }
            Self::ResourceLimitExceeded => {
                write!(
                    f,
                    "morphogenesis observation exceeds canonical digest resource limits"
                )
            }
        }
    }
}

impl Error for MycelialMorphogenesisObservationError {}

fn validate_identifier(
    value: &str,
    field: &'static str,
    maximum_bytes: usize,
) -> Result<(), MycelialMorphogenesisObservationError> {
    if value.trim().is_empty() || value.len() > maximum_bytes || value.chars().any(char::is_control)
    {
        return Err(MycelialMorphogenesisObservationError::InvalidIdentifier(
            field,
        ));
    }
    Ok(())
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialMorphogenesisObservationError> {
    if value.len() != DIGEST_HEX_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MycelialMorphogenesisObservationError::InvalidDigest(field));
    }
    Ok(())
}

fn validate_finite(
    value: f64,
    field: &'static str,
) -> Result<(), MycelialMorphogenesisObservationError> {
    if !value.is_finite() {
        return Err(MycelialMorphogenesisObservationError::InvalidMeasurement(
            field,
        ));
    }
    Ok(())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn region_kind_tag(kind: MycelialMorphogenesisRegionKind) -> u8 {
    match kind {
        MycelialMorphogenesisRegionKind::SubstratePatch => 0,
        MycelialMorphogenesisRegionKind::CordSegment => 1,
        MycelialMorphogenesisRegionKind::HyphalFront => 2,
        MycelialMorphogenesisRegionKind::Primordium => 3,
        MycelialMorphogenesisRegionKind::FruitingBody => 4,
    }
}

fn missing_status_tag(status: MycelialMorphogenesisMissingStatus) -> u8 {
    match status {
        MycelialMorphogenesisMissingStatus::NotAcquired => 0,
        MycelialMorphogenesisMissingStatus::UnavailableOrCorrupted => 1,
        MycelialMorphogenesisMissingStatus::Unmeasurable => 2,
        MycelialMorphogenesisMissingStatus::OccludedOrOutOfFocus => 3,
        MycelialMorphogenesisMissingStatus::BelowDetection => 4,
        MycelialMorphogenesisMissingStatus::RightCensored => 5,
        MycelialMorphogenesisMissingStatus::LeftCensored => 6,
        MycelialMorphogenesisMissingStatus::IntervalCensored => 7,
    }
}

fn development_stage_tag(stage: MycelialDevelopmentStage) -> u8 {
    match stage {
        MycelialDevelopmentStage::VegetativeColonization => 0,
        MycelialDevelopmentStage::CasingColonization => 1,
        MycelialDevelopmentStage::PrimordiumInitiation => 2,
        MycelialDevelopmentStage::ExponentialFruiting => 3,
        MycelialDevelopmentStage::Harvest => 4,
        MycelialDevelopmentStage::Unknown => 5,
    }
}

impl MycelialMorphogenesisObservation {
    pub fn validate(&self) -> Result<(), MycelialMorphogenesisObservationError> {
        if self.schema_version != MYCELIAL_MORPHOGENESIS_OBSERVATION_SCHEMA_VERSION {
            return Err(MycelialMorphogenesisObservationError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        for (value, field, maximum) in [
            (
                self.observation_id.as_str(),
                "observation_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.source_artifact_id.as_str(),
                "source_artifact_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.biological_replicate_id.as_str(),
                "biological_replicate_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.cultivation_unit_id.as_str(),
                "cultivation_unit_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.acquisition_session_id.as_str(),
                "acquisition_session_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.substrate_region_id.as_str(),
                "substrate_region_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.timebase_id.as_str(),
                "timebase_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
            (
                self.coordinate_frame_id.as_str(),
                "coordinate_frame_id",
                MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
            ),
        ] {
            validate_identifier(value, field, maximum)?;
        }
        for (value, field) in [
            (&self.source_observation_digest, "source_observation_digest"),
            (&self.source_artifact_digest, "source_artifact_digest"),
            (
                &self.acquisition_domain_manifest_digest,
                "acquisition_domain_manifest_digest",
            ),
            (
                &self.extraction_coverage_receipt_digest,
                "extraction_coverage_receipt_digest",
            ),
            (
                &self.observation_coverage_ledger_digest,
                "observation_coverage_ledger_digest",
            ),
            (&self.spatial_transform_digest, "spatial_transform_digest"),
            (&self.calibration_digest, "calibration_digest"),
        ] {
            validate_digest(value, field)?;
        }

        match &self.measurement {
            MycelialMorphogenesisMeasurement::MeasurementUnavailable { reason, .. } => {
                if reason.trim().is_empty()
                    || reason.len() > MAX_MORPHOGENESIS_IDENTIFIER_BYTES * 4
                    || reason.chars().any(char::is_control)
                {
                    return Err(MycelialMorphogenesisObservationError::InvalidMissingReason);
                }
            }
            MycelialMorphogenesisMeasurement::ColonizationDensityProxy {
                mean_intensity,
                scale_min,
                scale_max,
                intensity_scale_id,
                analysis_method_id,
            } => {
                for (value, name) in [
                    (*mean_intensity, "mean_intensity"),
                    (*scale_min, "scale_min"),
                    (*scale_max, "scale_max"),
                ] {
                    validate_finite(value, name)?;
                }
                if scale_min >= scale_max
                    || mean_intensity < scale_min
                    || mean_intensity > scale_max
                {
                    return Err(MycelialMorphogenesisObservationError::InvalidMeasurement(
                        "colonization density proxy range",
                    ));
                }
                validate_identifier(
                    intensity_scale_id,
                    "intensity_scale_id",
                    MAX_MORPHOGENESIS_IDENTIFIER_BYTES,
                )?;
                validate_identifier(
                    analysis_method_id,
                    "analysis_method_id",
                    MAX_MORPHOGENESIS_METHOD_BYTES,
                )?;
            }
            MycelialMorphogenesisMeasurement::CordAreaFraction {
                fraction,
                analysis_method_id,
            } => {
                validate_finite(*fraction, "cord_area_fraction")?;
                if !(0.0..=1.0).contains(fraction) {
                    return Err(MycelialMorphogenesisObservationError::InvalidMeasurement(
                        "cord area fraction must be in [0, 1]",
                    ));
                }
                validate_identifier(
                    analysis_method_id,
                    "analysis_method_id",
                    MAX_MORPHOGENESIS_METHOD_BYTES,
                )?;
            }
            MycelialMorphogenesisMeasurement::CordWidthDistribution {
                sample_count,
                q10,
                median,
                q90,
                unit,
                analysis_method_id,
            } => {
                for (value, name) in [
                    (*q10, "cord_width_q10"),
                    (*median, "cord_width_median"),
                    (*q90, "cord_width_q90"),
                ] {
                    validate_finite(value, name)?;
                }
                if *sample_count == 0 || *q10 <= 0.0 || q10 > median || median > q90 {
                    return Err(MycelialMorphogenesisObservationError::InvalidMeasurement(
                        "cord width sample count or quantile ordering",
                    ));
                }
                validate_identifier(unit, "cord_width_unit", MAX_MORPHOGENESIS_UNIT_BYTES)?;
                validate_identifier(
                    analysis_method_id,
                    "analysis_method_id",
                    MAX_MORPHOGENESIS_METHOD_BYTES,
                )?;
            }
            MycelialMorphogenesisMeasurement::DirectionalExtension {
                dx,
                dy,
                dz,
                unit,
                analysis_method_id,
            } => {
                for (value, name) in [
                    (*dx, "extension_dx"),
                    (*dy, "extension_dy"),
                    (*dz, "extension_dz"),
                ] {
                    validate_finite(value, name)?;
                }
                let magnitude = dx.hypot(*dy).hypot(*dz);
                if !magnitude.is_finite() || magnitude <= 0.0 {
                    return Err(MycelialMorphogenesisObservationError::InvalidMeasurement(
                        "directional extension vector must be finite and nonzero",
                    ));
                }
                validate_identifier(unit, "extension_unit", MAX_MORPHOGENESIS_UNIT_BYTES)?;
                validate_identifier(
                    analysis_method_id,
                    "analysis_method_id",
                    MAX_MORPHOGENESIS_METHOD_BYTES,
                )?;
            }
            MycelialMorphogenesisMeasurement::PrimordiumArea {
                area,
                unit,
                analysis_method_id,
            } => {
                validate_finite(*area, "primordium_area")?;
                if *area <= 0.0 {
                    return Err(MycelialMorphogenesisObservationError::InvalidMeasurement(
                        "primordium area must be positive",
                    ));
                }
                validate_identifier(unit, "primordium_area_unit", MAX_MORPHOGENESIS_UNIT_BYTES)?;
                validate_identifier(
                    analysis_method_id,
                    "analysis_method_id",
                    MAX_MORPHOGENESIS_METHOD_BYTES,
                )?;
            }
            MycelialMorphogenesisMeasurement::CordPresence { analysis_method_id, .. } => {
                validate_identifier(
                    analysis_method_id,
                    "analysis_method_id",
                    MAX_MORPHOGENESIS_METHOD_BYTES,
                )?;
            }
        }
        Ok(())
    }

    /// Bind this morphology observation to the complete extraction and observation evidence.
    ///
    /// Validation establishes provenance consistency, not causal interpretation:
    /// stage labels and observed morphology remain descriptive variables.
    pub fn validate_against_evidence(
        &self,
        manifest: &MycelialAcquisitionDomainManifest,
        receipt: &MycelialExtractionCoverageReceipt,
        ledger: &MycelialObservationCoverageLedger,
    ) -> Result<(), MycelialMorphogenesisObservationError> {
        self.validate()?;
        receipt
            .validate_observation_ledger(manifest, ledger)
            .map_err(|_| {
                MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                    "acquisition, extraction and observation inventory",
                )
            })?;

        if self.source_artifact_id != manifest.source_artifact_id
            || self.source_artifact_digest != manifest.source_artifact_digest
            || self.acquisition_domain_manifest_digest
                != manifest.canonical_digest_hex().map_err(|_| {
                    MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                        "acquisition manifest digest",
                    )
                })?
            || self.extraction_coverage_receipt_digest
                != receipt.canonical_digest_hex().map_err(|_| {
                    MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                        "extraction receipt digest",
                    )
                })?
            || self.observation_coverage_ledger_digest
                != ledger.canonical_digest_hex().map_err(|_| {
                    MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                        "observation ledger digest",
                    )
                })?
            || self.biological_replicate_id != manifest.biological_replicate_id
            || self.acquisition_session_id != manifest.acquisition_session_id
            || self.coordinate_frame_id != manifest.coordinate_frame_id
            || self.spatial_transform_digest != manifest.spatial_transform_digest
            || self.calibration_digest != manifest.calibration_digest
        {
            return Err(
                MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                    "source, replicate, acquisition or coordinate identity",
                ),
            );
        }

        let Some(record) = ledger
            .records
            .binary_search_by(|record| record.observation_id.cmp(&self.observation_id))
            .ok()
            .and_then(|index| ledger.records.get(index))
        else {
            return Err(MycelialMorphogenesisObservationError::ObservationNotInLedger);
        };
        if record.observation_digest != self.source_observation_digest {
            return Err(
                MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                    "source observation digest",
                ),
            );
        }
        if record.source_unit_key != Some(self.source_unit_key) {
            return Err(MycelialMorphogenesisObservationError::SourceUnitMismatch);
        }
        Ok(())
    }

    /// Canonical content digest over exact source bindings and typed measurement values.
    pub fn canonical_digest_hex(&self) -> Result<String, MycelialMorphogenesisObservationError> {
        if self.observation_id.len() + self.substrate_region_id.len()
            > MAX_MORPHOGENESIS_STRING_BYTES
        {
            return Err(MycelialMorphogenesisObservationError::ResourceLimitExceeded);
        }
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-morphogenesis-observation-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        for value in [
            self.observation_id.as_str(),
            self.source_observation_digest.as_str(),
            self.source_artifact_id.as_str(),
            self.source_artifact_digest.as_str(),
            self.acquisition_domain_manifest_digest.as_str(),
            self.extraction_coverage_receipt_digest.as_str(),
            self.observation_coverage_ledger_digest.as_str(),
            self.biological_replicate_id.as_str(),
            self.cultivation_unit_id.as_str(),
            self.acquisition_session_id.as_str(),
            self.substrate_region_id.as_str(),
            self.timebase_id.as_str(),
            self.coordinate_frame_id.as_str(),
            self.spatial_transform_digest.as_str(),
            self.calibration_digest.as_str(),
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&[region_kind_tag(self.region_kind)]);
        hasher.update(&self.source_unit_key.tile_x.to_le_bytes());
        hasher.update(&self.source_unit_key.tile_y.to_le_bytes());
        hasher.update(&self.source_unit_key.z_plane.to_le_bytes());
        hasher.update(&self.source_unit_key.channel.to_le_bytes());
        hasher.update(&self.source_unit_key.timepoint.to_le_bytes());
        hasher.update(&self.timestamp_ns.to_le_bytes());
        hasher.update(&[development_stage_tag(self.development_stage)]);

        match &self.measurement {
            MycelialMorphogenesisMeasurement::MeasurementUnavailable { status, reason } => {
                hasher.update(&[6, missing_status_tag(*status)]);
                hash_string(&mut hasher, reason);
            }
            MycelialMorphogenesisMeasurement::ColonizationDensityProxy {
                mean_intensity,
                scale_min,
                scale_max,
                intensity_scale_id,
                analysis_method_id,
            } => {
                hasher.update(&[0]);
                hash_f64(&mut hasher, *mean_intensity);
                hash_f64(&mut hasher, *scale_min);
                hash_f64(&mut hasher, *scale_max);
                hash_string(&mut hasher, intensity_scale_id);
                hash_string(&mut hasher, analysis_method_id);
            }
            MycelialMorphogenesisMeasurement::CordAreaFraction {
                fraction,
                analysis_method_id,
            } => {
                hasher.update(&[1]);
                hash_f64(&mut hasher, *fraction);
                hash_string(&mut hasher, analysis_method_id);
            }
            MycelialMorphogenesisMeasurement::CordWidthDistribution {
                sample_count,
                q10,
                median,
                q90,
                unit,
                analysis_method_id,
            } => {
                hasher.update(&[2]);
                hasher.update(&sample_count.to_le_bytes());
                hash_f64(&mut hasher, *q10);
                hash_f64(&mut hasher, *median);
                hash_f64(&mut hasher, *q90);
                hash_string(&mut hasher, unit);
                hash_string(&mut hasher, analysis_method_id);
            }
            MycelialMorphogenesisMeasurement::DirectionalExtension {
                dx,
                dy,
                dz,
                unit,
                analysis_method_id,
            } => {
                hasher.update(&[3]);
                hash_f64(&mut hasher, *dx);
                hash_f64(&mut hasher, *dy);
                hash_f64(&mut hasher, *dz);
                hash_string(&mut hasher, unit);
                hash_string(&mut hasher, analysis_method_id);
            }
            MycelialMorphogenesisMeasurement::PrimordiumArea {
                area,
                unit,
                analysis_method_id,
            } => {
                hasher.update(&[4]);
                hash_f64(&mut hasher, *area);
                hash_string(&mut hasher, unit);
                hash_string(&mut hasher, analysis_method_id);
            }
            MycelialMorphogenesisMeasurement::CordPresence {
                present,
                analysis_method_id,
            } => {
                hasher.update(&[5, u8::from(*present)]);
                hash_string(&mut hasher, analysis_method_id);
            }
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, DIGEST_HEX_BYTES).collect()
    }

    fn fixture(measurement: MycelialMorphogenesisMeasurement) -> MycelialMorphogenesisObservation {
        MycelialMorphogenesisObservation {
            schema_version: MYCELIAL_MORPHOGENESIS_OBSERVATION_SCHEMA_VERSION,
            observation_id: "morph-001".to_string(),
            source_observation_digest: digest('1'),
            source_artifact_id: "source-001".to_string(),
            source_artifact_digest: digest('a'),
            acquisition_domain_manifest_digest: digest('b'),
            extraction_coverage_receipt_digest: digest('c'),
            observation_coverage_ledger_digest: digest('d'),
            biological_replicate_id: "replicate-001".to_string(),
            cultivation_unit_id: "box-001".to_string(),
            acquisition_session_id: "session-001".to_string(),
            substrate_region_id: "substrate-region-001".to_string(),
            region_kind: MycelialMorphogenesisRegionKind::SubstratePatch,
            source_unit_key: MycelialAcquisitionUnitKey {
                tile_x: 0,
                tile_y: 0,
                z_plane: 0,
                channel: 0,
                timepoint: 0,
            },
            timestamp_ns: 1_000_000,
            timebase_id: "monotonic-ns-v1".to_string(),
            development_stage: MycelialDevelopmentStage::VegetativeColonization,
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            spatial_transform_digest: digest('e'),
            calibration_digest: digest('f'),
            measurement,
        }
    }

    fn evidence_fixture() -> (
        MycelialAcquisitionDomainManifest,
        MycelialExtractionCoverageReceipt,
        MycelialObservationCoverageLedger,
        MycelialMorphogenesisObservation,
    ) {
        use crate::mycelial_acquisition_domain::{
            MYCELIAL_ACQUISITION_DOMAIN_SCHEMA_VERSION, MYCELIAL_EXTRACTION_RECEIPT_SCHEMA_VERSION,
            MycelialAcquisitionAxis, MycelialAcquisitionAxisName, MycelialAcquisitionDomainSpec,
            MycelialExtractionUnitOutcome, MycelialExtractionUnitState, MycelialSourceRightsStatus,
        };
        use crate::mycelial_observation_coverage::{
            MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION, MycelialObservationCoverageRecord,
            MycelialObservationCoverageState, mycelial_observation_id_inventory_digest,
        };

        let unit = MycelialAcquisitionUnitKey {
            tile_x: 0,
            tile_y: 0,
            z_plane: 0,
            channel: 0,
            timepoint: 0,
        };
        let manifest = MycelialAcquisitionDomainManifest {
            schema_version: MYCELIAL_ACQUISITION_DOMAIN_SCHEMA_VERSION,
            source_artifact_id: "source-001".to_string(),
            source_artifact_digest: digest('a'),
            source_rights_status: MycelialSourceRightsStatus::VerifiedForProcessing,
            rights_evidence_digest: digest('b'),
            source_format: "ome_tiff".to_string(),
            acquisition_session_id: "session-001".to_string(),
            biological_replicate_id: "replicate-001".to_string(),
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            declared_scope_digest: digest('c'),
            instrument_metadata_digest: digest('d'),
            axes: vec![
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::X,
                    length: 1,
                    unit: "pixel".to_string(),
                    origin: 0.0,
                    physical_scale: Some(0.2),
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Y,
                    length: 1,
                    unit: "pixel".to_string(),
                    origin: 0.0,
                    physical_scale: Some(0.2),
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Z,
                    length: 1,
                    unit: "micrometer".to_string(),
                    origin: 0.0,
                    physical_scale: Some(0.5),
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Channel,
                    length: 1,
                    unit: "channel_index".to_string(),
                    origin: 0.0,
                    physical_scale: None,
                },
                MycelialAcquisitionAxis {
                    name: MycelialAcquisitionAxisName::Time,
                    length: 1,
                    unit: "frame".to_string(),
                    origin: 0.0,
                    physical_scale: None,
                },
            ],
            domain: MycelialAcquisitionDomainSpec {
                width_pixels: 1,
                height_pixels: 1,
                z_planes: 1,
                channels: 1,
                timepoints: 1,
                tile_width_pixels: 1,
                tile_height_pixels: 1,
            },
            spatial_transform_digest: digest('e'),
            calibration_digest: digest('f'),
            manifest_producer_id: "fixture-reader".to_string(),
            manifest_producer_version: "1.0.0".to_string(),
            manifest_producer_digest: digest('1'),
            scope_exclusions: Vec::new(),
        };
        manifest.validate().unwrap();

        let observation_ids = vec!["morph-001".to_string()];
        let id_digest = mycelial_observation_id_inventory_digest(&observation_ids);
        let receipt = MycelialExtractionCoverageReceipt {
            schema_version: MYCELIAL_EXTRACTION_RECEIPT_SCHEMA_VERSION,
            acquisition_domain_manifest_digest: manifest.canonical_digest_hex().unwrap(),
            extractor_id: "morphogenesis-test-extractor".to_string(),
            extractor_version: "0.1.0".to_string(),
            extractor_config_digest: digest('2'),
            units: vec![MycelialExtractionUnitOutcome {
                unit,
                source_unit_digest: digest('3'),
                state: MycelialExtractionUnitState::Processed,
                output_observation_inventory_digest: Some(id_digest.clone()),
                output_observation_count: Some(1),
                disposition_reason: None,
            }],
        };

        let ledger = MycelialObservationCoverageLedger {
            schema_version: MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION,
            source_dataset_id: manifest.source_artifact_id.clone(),
            source_dataset_digest: manifest.source_artifact_digest.clone(),
            observation_inventory_digest: id_digest,
            declared_observation_ids: observation_ids,
            coordinate_frame_id: manifest.coordinate_frame_id.clone(),
            spatial_transform_digest: manifest.spatial_transform_digest.clone(),
            calibration_digest: manifest.calibration_digest.clone(),
            candidate_generator_id: "fixture-generator".to_string(),
            candidate_generator_version: "1.0.0".to_string(),
            candidate_generator_digest: digest('4'),
            records: vec![MycelialObservationCoverageRecord {
                observation_id: "morph-001".to_string(),
                observation_digest: digest('5'),
                source_unit_key: Some(unit),
                acquisition_group_id: manifest.acquisition_session_id.clone(),
                biological_replicate_id: manifest.biological_replicate_id.clone(),
                time_window_id: "time-000".to_string(),
                state: MycelialObservationCoverageState::CandidateEmitted,
                candidate_ids: vec!["candidate-001".to_string()],
                residual_kind: None,
            }],
        };

        let mut observation = fixture(MycelialMorphogenesisMeasurement::CordAreaFraction {
            fraction: 0.25,
            analysis_method_id: "cord-segmentation-v1".to_string(),
        });
        observation.source_observation_digest = digest('5');
        observation.source_artifact_id = manifest.source_artifact_id.clone();
        observation.source_artifact_digest = manifest.source_artifact_digest.clone();
        observation.acquisition_domain_manifest_digest = manifest.canonical_digest_hex().unwrap();
        observation.extraction_coverage_receipt_digest = receipt.canonical_digest_hex().unwrap();
        observation.observation_coverage_ledger_digest = ledger.canonical_digest_hex().unwrap();
        observation.biological_replicate_id = manifest.biological_replicate_id.clone();
        observation.acquisition_session_id = manifest.acquisition_session_id.clone();
        observation.source_unit_key = unit;
        observation.coordinate_frame_id = manifest.coordinate_frame_id.clone();
        observation.spatial_transform_digest = manifest.spatial_transform_digest.clone();
        observation.calibration_digest = manifest.calibration_digest.clone();

        (manifest, receipt, ledger, observation)
    }

    #[test]
    fn missing_and_censored_measurements_remain_distinct_from_zero() {
        let unavailable = fixture(MycelialMorphogenesisMeasurement::MeasurementUnavailable {
            status: MycelialMorphogenesisMissingStatus::BelowDetection,
            reason: "cord segmentation below validated detection threshold".to_string(),
        });
        let zero_area = fixture(MycelialMorphogenesisMeasurement::CordAreaFraction {
            fraction: 0.0,
            analysis_method_id: "cord-segmentation-v1".to_string(),
        });
        unavailable.validate().unwrap();
        zero_area.validate().unwrap();
        assert_ne!(
            unavailable.canonical_digest_hex().unwrap(),
            zero_area.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn missing_measurement_requires_a_reason() {
        let observation = fixture(MycelialMorphogenesisMeasurement::MeasurementUnavailable {
            status: MycelialMorphogenesisMissingStatus::OccludedOrOutOfFocus,
            reason: "  ".to_string(),
        });
        assert_eq!(
            observation.validate(),
            Err(MycelialMorphogenesisObservationError::InvalidMissingReason)
        );
    }

    #[test]
    fn observation_validates_against_complete_acquisition_evidence() {
        let (manifest, receipt, ledger, observation) = evidence_fixture();
        observation
            .validate_against_evidence(&manifest, &receipt, &ledger)
            .unwrap();
    }

    #[test]
    fn observation_source_digest_must_match_ledger_record() {
        let (manifest, receipt, ledger, mut observation) = evidence_fixture();
        observation.source_observation_digest = digest('9');
        assert!(matches!(
            observation.validate_against_evidence(&manifest, &receipt, &ledger),
            Err(
                MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                    "source observation digest"
                )
            )
        ));
    }

    #[test]
    fn observation_missing_ledger_unit_binding_fails_closed() {
        let (manifest, receipt, mut ledger, mut observation) = evidence_fixture();
        ledger.records[0].source_unit_key = None;
        observation.observation_coverage_ledger_digest = ledger.canonical_digest_hex().unwrap();
        assert!(matches!(
            observation.validate_against_evidence(&manifest, &receipt, &ledger),
            Err(
                MycelialMorphogenesisObservationError::EvidenceChainMismatch(
                    "acquisition, extraction and observation inventory"
                )
            )
        ));
    }

    #[test]
    fn grayscale_density_is_explicitly_a_proxy_and_validates() {
        let observation = fixture(MycelialMorphogenesisMeasurement::ColonizationDensityProxy {
                mean_intensity: 0.4,
                scale_min: 0.0,
                scale_max: 1.0,
                intensity_scale_id: "normalized-grayscale-v1".to_string(),
            analysis_method_id: "segmentation-v1".to_string(),
        });
        observation.validate().unwrap();
        assert_eq!(observation.canonical_digest_hex().unwrap().len(), 64);
    }

    #[test]
    fn grayscale_proxy_outside_declared_scale_is_rejected() {
        let observation = fixture(
            MycelialMorphogenesisMeasurement::ColonizationDensityProxy {
                mean_intensity: 1.2,
                scale_min: 0.0,
                scale_max: 1.0,
                intensity_scale_id: "normalized-grayscale-v1".to_string(),
                analysis_method_id: "segmentation-v1".to_string(),
            },
        );
        assert!(matches!(
            observation.validate(),
            Err(MycelialMorphogenesisObservationError::InvalidMeasurement(_))
        ));
    }

    #[test]
    fn cord_area_fraction_is_bounded() {
        let valid = fixture(MycelialMorphogenesisMeasurement::CordAreaFraction {
            fraction: 0.25,
            analysis_method_id: "cord-segmentation-v1".to_string(),
        });
        valid.validate().unwrap();

        let invalid = fixture(MycelialMorphogenesisMeasurement::CordAreaFraction {
            fraction: 1.5,
            analysis_method_id: "cord-segmentation-v1".to_string(),
        });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn cord_width_distribution_requires_ordered_positive_quantiles() {
        let valid = fixture(MycelialMorphogenesisMeasurement::CordWidthDistribution {
            sample_count: 20,
            q10: 2.0,
            median: 4.0,
            q90: 7.0,
            unit: "micrometer".to_string(),
            analysis_method_id: "cord-width-v1".to_string(),
        });
        valid.validate().unwrap();

        let invalid = fixture(MycelialMorphogenesisMeasurement::CordWidthDistribution {
            sample_count: 20,
            q10: 5.0,
            median: 4.0,
            q90: 7.0,
            unit: "micrometer".to_string(),
            analysis_method_id: "cord-width-v1".to_string(),
        });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn directional_extension_requires_nonzero_finite_vector() {
        let valid = fixture(MycelialMorphogenesisMeasurement::DirectionalExtension {
            dx: 1.0,
            dy: 0.0,
            dz: 0.0,
            unit: "micrometer".to_string(),
            analysis_method_id: "tip-tracking-v1".to_string(),
        });
        valid.validate().unwrap();

        let invalid = fixture(MycelialMorphogenesisMeasurement::DirectionalExtension {
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
            unit: "micrometer".to_string(),
            analysis_method_id: "tip-tracking-v1".to_string(),
        });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn primordium_area_must_be_positive() {
        let invalid = fixture(MycelialMorphogenesisMeasurement::PrimordiumArea {
            area: 0.0,
            unit: "square_micrometer".to_string(),
            analysis_method_id: "primordium-segmentation-v1".to_string(),
        });
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn measurement_type_changes_content_digest() {
        let proxy = fixture(MycelialMorphogenesisMeasurement::ColonizationDensityProxy {
            mean_intensity: 0.4,
            scale_min: 0.0,
            scale_max: 1.0,
            intensity_scale_id: "normalized-grayscale-v1".to_string(),
            analysis_method_id: "segmentation-v1".to_string(),
        });
        let cord_area = fixture(MycelialMorphogenesisMeasurement::CordAreaFraction {
            fraction: 0.4,
            analysis_method_id: "cord-segmentation-v1".to_string(),
        });
        assert_ne!(
            proxy.canonical_digest_hex().unwrap(),
            cord_area.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn biological_unit_and_timepoint_are_committed() {
        let original = fixture(MycelialMorphogenesisMeasurement::CordPresence {
            present: true,
            analysis_method_id: "manual-adjudication-v1".to_string(),
        });
        let digest = original.canonical_digest_hex().unwrap();
        let mut changed = original;
        changed.cultivation_unit_id = "box-002".to_string();
        assert_ne!(changed.canonical_digest_hex().unwrap(), digest);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let observation = fixture(MycelialMorphogenesisMeasurement::CordPresence {
            present: true,
            analysis_method_id: "manual-adjudication-v1".to_string(),
        });
        let mut value = serde_json::to_value(observation).unwrap();
        value["unbound_causal_claim"] = serde_json::json!("demand caused cord growth");
        assert!(serde_json::from_value::<MycelialMorphogenesisObservation>(value).is_err());
    }
}
