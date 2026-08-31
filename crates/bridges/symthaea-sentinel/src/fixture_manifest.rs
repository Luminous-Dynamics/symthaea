//! Content-addressed Sentinel fixture manifests for reproducible Earth-observation research.
//!
//! The fixture layer freezes provider metadata and materially transformed artifacts. It does not
//! own train/calibration/evaluation assignment or access custody; those remain separate research
//! integrity concerns.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use symthaea_earth_observation::{
    ContentDigest, DigestAlgorithm, Polarization, ProcessingStep, RadarBand, RadiometricUnit,
    SensorModality,
};
use thiserror::Error;

use crate::{SentinelMission, SentinelProductKind, SentinelProductMetadata};

const FIXTURE_SCHEMA: &str = "symthaea-sentinel-fixture-manifest/v1";
const PRODUCT_SCHEMA: &str = "symthaea-sentinel-frozen-product/v1";
const ARTIFACT_SCHEMA: &str = "symthaea-sentinel-derived-artifact/v1";

pub type FixtureResult<T> = std::result::Result<T, FixtureError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FixtureError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("invalid or noncanonical hexadecimal digest in {0}")]
    InvalidDigest(&'static str),
    #[error("unsupported or noncanonical digest algorithm: {0}")]
    InvalidDigestAlgorithm(String),
    #[error("invalid frozen floating-point field: {0}")]
    InvalidFloat(&'static str),
    #[error("a frozen footprint requires at least three vertices, got {0}")]
    InvalidFootprint(usize),
    #[error("Sentinel mission/product-kind mismatch for {0}")]
    MissionProductMismatch(String),
    #[error("duplicate Sentinel product id: {0}")]
    DuplicateProduct(String),
    #[error("duplicate derived artifact id: {0}")]
    DuplicateArtifact(String),
    #[error("fixture identifier is reused by both a product and artifact: {0}")]
    IdentifierCollision(String),
    #[error("derived artifact {artifact_id} has duplicate source reference {source_id}")]
    DuplicateSource { artifact_id: String, source_id: String },
    #[error("derived artifact {0} requires at least one source")]
    MissingArtifactSource(String),
    #[error("derived artifact {0} requires at least one processing step")]
    MissingProcessingStep(String),
    #[error("derived artifact {artifact_id} references unknown product {product_id}")]
    UnknownProductSource { artifact_id: String, product_id: String },
    #[error("derived artifact {artifact_id} references unknown artifact {source_artifact_id}")]
    UnknownArtifactSource {
        artifact_id: String,
        source_artifact_id: String,
    },
    #[error("derived artifact lineage contains a cycle at {0}")]
    ArtifactCycle(String),
    #[error("stored product metadata digest does not match recomputation for {0}")]
    ProductDigestMismatch(String),
    #[error("stored artifact identity digest does not match recomputation for {0}")]
    ArtifactDigestMismatch(String),
    #[error("stored fixture manifest digest does not match recomputation")]
    ManifestDigestMismatch,
    #[error("fixture serialization failed: {0}")]
    Serialization(String),
}

fn non_empty(value: &str, field: &'static str) -> FixtureResult<()> {
    if value.trim().is_empty() {
        Err(FixtureError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_hex(value: &str, field: &'static str) -> FixtureResult<()> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        Err(FixtureError::InvalidDigest(field))
    } else {
        Ok(())
    }
}

fn blake3_json<T: Serialize>(value: &T) -> FixtureResult<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| FixtureError::Serialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenDigest {
    pub algorithm: String,
    pub hex: String,
}

impl FrozenDigest {
    pub fn from_content_digest(value: &ContentDigest) -> Self {
        let algorithm = match value.algorithm {
            DigestAlgorithm::Sha256 => "sha256",
            DigestAlgorithm::Blake3 => "blake3",
            DigestAlgorithm::Other => "other",
        };
        Self {
            algorithm: algorithm.to_string(),
            hex: value.hex.to_ascii_lowercase(),
        }
    }

    fn validate(&self) -> FixtureResult<()> {
        match self.algorithm.as_str() {
            "sha256" | "blake3" | "other" => {}
            _ => return Err(FixtureError::InvalidDigestAlgorithm(self.algorithm.clone())),
        }
        validate_hex(&self.hex, "content digest")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrozenSentinelMission {
    Sentinel1,
    Sentinel2,
}

impl From<SentinelMission> for FrozenSentinelMission {
    fn from(value: SentinelMission) -> Self {
        match value {
            SentinelMission::Sentinel1 => Self::Sentinel1,
            SentinelMission::Sentinel2 => Self::Sentinel2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrozenSentinelProductKind {
    Sentinel1Grd,
    Sentinel1Slc,
    Sentinel2L1C,
    Sentinel2L2A,
}

impl FrozenSentinelProductKind {
    fn mission(self) -> FrozenSentinelMission {
        match self {
            Self::Sentinel1Grd | Self::Sentinel1Slc => FrozenSentinelMission::Sentinel1,
            Self::Sentinel2L1C | Self::Sentinel2L2A => FrozenSentinelMission::Sentinel2,
        }
    }
}

impl From<SentinelProductKind> for FrozenSentinelProductKind {
    fn from(value: SentinelProductKind) -> Self {
        match value {
            SentinelProductKind::Sentinel1Grd => Self::Sentinel1Grd,
            SentinelProductKind::Sentinel1Slc => Self::Sentinel1Slc,
            SentinelProductKind::Sentinel2L1C => Self::Sentinel2L1C,
            SentinelProductKind::Sentinel2L2A => Self::Sentinel2L2A,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenGeoPoint {
    /// Exact IEEE-754 bits from the validated provider-neutral latitude.
    pub latitude_bits: u64,
    /// Exact IEEE-754 bits from the validated provider-neutral longitude.
    pub longitude_bits: u64,
}

impl FrozenGeoPoint {
    fn validate(&self) -> FixtureResult<()> {
        let latitude = f64::from_bits(self.latitude_bits);
        let longitude = f64::from_bits(self.longitude_bits);
        if !latitude.is_finite() || !(-90.0..=90.0).contains(&latitude) {
            return Err(FixtureError::InvalidFloat("latitude"));
        }
        if !longitude.is_finite() || !(-180.0..=180.0).contains(&longitude) {
            return Err(FixtureError::InvalidFloat("longitude"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenBand {
    pub name: String,
    pub center_wavelength_nm_bits: Option<u64>,
    pub unit: String,
}

impl FrozenBand {
    fn validate(&self) -> FixtureResult<()> {
        non_empty(&self.name, "band name")?;
        non_empty(&self.unit, "radiometric unit")?;
        if let Some(bits) = self.center_wavelength_nm_bits {
            let value = f64::from_bits(bits);
            if !value.is_finite() || value < 0.0 {
                return Err(FixtureError::InvalidFloat("center wavelength"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenUncertainty {
    pub confidence_bits: Option<u64>,
    pub standard_uncertainty_bits: Option<u64>,
    pub note: Option<String>,
}

impl FrozenUncertainty {
    fn validate(&self) -> FixtureResult<()> {
        if let Some(bits) = self.confidence_bits {
            let value = f64::from_bits(bits);
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(FixtureError::InvalidFloat("confidence"));
            }
        }
        if let Some(bits) = self.standard_uncertainty_bits {
            let value = f64::from_bits(bits);
            if !value.is_finite() || value < 0.0 {
                return Err(FixtureError::InvalidFloat("standard uncertainty"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenProcessingStep {
    pub name: String,
    pub software: String,
    pub version: String,
    pub parameters_digest: Option<FrozenDigest>,
}

impl FrozenProcessingStep {
    pub fn from_processing_step(step: &ProcessingStep) -> Self {
        Self {
            name: step.name.clone(),
            software: step.software.clone(),
            version: step.version.clone(),
            parameters_digest: step
                .parameters_digest
                .as_ref()
                .map(FrozenDigest::from_content_digest),
        }
    }

    fn validate(&self) -> FixtureResult<()> {
        non_empty(&self.name, "processing step name")?;
        non_empty(&self.software, "processing step software")?;
        non_empty(&self.version, "processing step version")?;
        if let Some(digest) = &self.parameters_digest {
            digest.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FrozenSensorModality {
    Optical,
    Multispectral,
    Hyperspectral,
    SyntheticApertureRadar {
        band: String,
        polarization: Option<String>,
    },
    ThermalInfrared,
    Lidar,
    Gravity,
    Magnetics,
    AirborneElectromagnetic,
    GroundPenetratingRadar,
    ElectricalResistivity,
    Seismic,
    InSitu,
    Other { label: String },
}

impl FrozenSensorModality {
    fn from_sensor_modality(modality: &SensorModality) -> Self {
        match modality {
            SensorModality::Optical => Self::Optical,
            SensorModality::Multispectral => Self::Multispectral,
            SensorModality::Hyperspectral => Self::Hyperspectral,
            SensorModality::SyntheticApertureRadar { band, polarization } => {
                Self::SyntheticApertureRadar {
                    band: radar_band_tag(*band).to_string(),
                    polarization: polarization.map(|value| polarization_tag(value).to_string()),
                }
            }
            SensorModality::ThermalInfrared => Self::ThermalInfrared,
            SensorModality::Lidar => Self::Lidar,
            SensorModality::Gravity => Self::Gravity,
            SensorModality::Magnetics => Self::Magnetics,
            SensorModality::AirborneElectromagnetic => Self::AirborneElectromagnetic,
            SensorModality::GroundPenetratingRadar => Self::GroundPenetratingRadar,
            SensorModality::ElectricalResistivity => Self::ElectricalResistivity,
            SensorModality::Seismic => Self::Seismic,
            SensorModality::InSitu => Self::InSitu,
            SensorModality::Other(label) => Self::Other {
                label: label.clone(),
            },
        }
    }

    fn validate(&self) -> FixtureResult<()> {
        match self {
            Self::SyntheticApertureRadar { band, polarization } => {
                non_empty(band, "radar band")?;
                if let Some(value) = polarization {
                    non_empty(value, "radar polarization")?;
                }
            }
            Self::Other { label } => non_empty(label, "sensor modality label")?,
            _ => {}
        }
        Ok(())
    }
}

fn radar_band_tag(band: RadarBand) -> &'static str {
    match band {
        RadarBand::P => "p",
        RadarBand::L => "l",
        RadarBand::S => "s",
        RadarBand::C => "c",
        RadarBand::X => "x",
        RadarBand::Ku => "ku",
        RadarBand::Ka => "ka",
        RadarBand::Other => "other",
    }
}

fn polarization_tag(value: Polarization) -> &'static str {
    match value {
        Polarization::Hh => "hh",
        Polarization::Hv => "hv",
        Polarization::Vh => "vh",
        Polarization::Vv => "vv",
        Polarization::Circular => "circular",
        Polarization::Other => "other",
    }
}

fn radiometric_unit_tag(unit: RadiometricUnit) -> &'static str {
    match unit {
        RadiometricUnit::Reflectance => "reflectance",
        RadiometricUnit::BrightnessTemperature => "brightness-temperature",
        RadiometricUnit::BackscatterLinear => "backscatter-linear",
        RadiometricUnit::BackscatterDecibel => "backscatter-decibel",
        RadiometricUnit::Radiance => "radiance",
        RadiometricUnit::Dimensionless => "dimensionless",
        RadiometricUnit::Other => "other",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenSentinelProduct {
    pub observation_id: String,
    pub mission_id: String,
    pub mission: FrozenSentinelMission,
    pub instrument_id: String,
    pub product_id: String,
    pub product_kind: FrozenSentinelProductKind,
    pub acquired_at_unix_ms: i64,
    pub footprint: Vec<FrozenGeoPoint>,
    pub modality: FrozenSensorModality,
    pub bands: Vec<FrozenBand>,
    pub uncertainty: FrozenUncertainty,
    pub source_digest: FrozenDigest,
    pub processing_lineage: Vec<FrozenProcessingStep>,
    pub metadata_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct FrozenSentinelProductRepr {
    observation_id: String,
    mission_id: String,
    mission: FrozenSentinelMission,
    instrument_id: String,
    product_id: String,
    product_kind: FrozenSentinelProductKind,
    acquired_at_unix_ms: i64,
    footprint: Vec<FrozenGeoPoint>,
    modality: FrozenSensorModality,
    bands: Vec<FrozenBand>,
    uncertainty: FrozenUncertainty,
    source_digest: FrozenDigest,
    processing_lineage: Vec<FrozenProcessingStep>,
    metadata_digest: String,
}

#[derive(Serialize)]
struct ProductDigestView<'a> {
    schema: &'static str,
    observation_id: &'a str,
    mission_id: &'a str,
    mission: FrozenSentinelMission,
    instrument_id: &'a str,
    product_id: &'a str,
    product_kind: FrozenSentinelProductKind,
    acquired_at_unix_ms: i64,
    footprint: &'a [FrozenGeoPoint],
    modality: &'a FrozenSensorModality,
    bands: &'a [FrozenBand],
    uncertainty: &'a FrozenUncertainty,
    source_digest: &'a FrozenDigest,
    processing_lineage: &'a [FrozenProcessingStep],
}

impl FrozenSentinelProduct {
    pub fn from_metadata(product: &SentinelProductMetadata) -> FixtureResult<Self> {
        let mut result = Self {
            observation_id: product.observation_id.clone(),
            mission_id: product.mission_id.clone(),
            mission: product.product_kind.mission().into(),
            instrument_id: product.instrument_id.clone(),
            product_id: product.product_id.clone(),
            product_kind: product.product_kind.into(),
            acquired_at_unix_ms: product.acquired_at_unix_ms,
            footprint: product
                .footprint
                .vertices
                .iter()
                .map(|point| FrozenGeoPoint {
                    latitude_bits: point.latitude_deg.to_bits(),
                    longitude_bits: point.longitude_deg.to_bits(),
                })
                .collect(),
            modality: FrozenSensorModality::from_sensor_modality(&product.modality),
            bands: product
                .bands
                .iter()
                .map(|band| FrozenBand {
                    name: band.name.clone(),
                    center_wavelength_nm_bits: band.center_wavelength_nm.map(f64::to_bits),
                    unit: radiometric_unit_tag(band.unit).to_string(),
                })
                .collect(),
            uncertainty: FrozenUncertainty {
                confidence_bits: product
                    .uncertainty
                    .confidence
                    .map(|value| value.get().to_bits()),
                standard_uncertainty_bits: product
                    .uncertainty
                    .standard_uncertainty
                    .map(f64::to_bits),
                note: product.uncertainty.note.clone(),
            },
            source_digest: FrozenDigest::from_content_digest(&product.source_digest),
            processing_lineage: product
                .lineage
                .steps
                .iter()
                .map(FrozenProcessingStep::from_processing_step)
                .collect(),
            metadata_digest: String::new(),
        };
        result.validate_payload()?;
        result.metadata_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> ProductDigestView<'_> {
        ProductDigestView {
            schema: PRODUCT_SCHEMA,
            observation_id: &self.observation_id,
            mission_id: &self.mission_id,
            mission: self.mission,
            instrument_id: &self.instrument_id,
            product_id: &self.product_id,
            product_kind: self.product_kind,
            acquired_at_unix_ms: self.acquired_at_unix_ms,
            footprint: &self.footprint,
            modality: &self.modality,
            bands: &self.bands,
            uncertainty: &self.uncertainty,
            source_digest: &self.source_digest,
            processing_lineage: &self.processing_lineage,
        }
    }

    pub fn compute_digest(&self) -> FixtureResult<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> FixtureResult<()> {
        self.validate_payload()?;
        validate_hex(&self.metadata_digest, "product metadata digest")?;
        if self.compute_digest()? != self.metadata_digest {
            return Err(FixtureError::ProductDigestMismatch(self.product_id.clone()));
        }
        Ok(())
    }

    fn validate_payload(&self) -> FixtureResult<()> {
        non_empty(&self.observation_id, "observation id")?;
        non_empty(&self.mission_id, "mission id")?;
        non_empty(&self.instrument_id, "instrument id")?;
        non_empty(&self.product_id, "product id")?;
        if self.product_kind.mission() != self.mission {
            return Err(FixtureError::MissionProductMismatch(self.product_id.clone()));
        }
        if self.footprint.len() < 3 {
            return Err(FixtureError::InvalidFootprint(self.footprint.len()));
        }
        for point in &self.footprint {
            point.validate()?;
        }
        self.modality.validate()?;
        for band in &self.bands {
            band.validate()?;
        }
        self.uncertainty.validate()?;
        self.source_digest.validate()?;
        for step in &self.processing_lineage {
            step.validate()?;
        }
        Ok(())
    }
}

impl TryFrom<FrozenSentinelProductRepr> for FrozenSentinelProduct {
    type Error = FixtureError;

    fn try_from(value: FrozenSentinelProductRepr) -> FixtureResult<Self> {
        let product = Self {
            observation_id: value.observation_id,
            mission_id: value.mission_id,
            mission: value.mission,
            instrument_id: value.instrument_id,
            product_id: value.product_id,
            product_kind: value.product_kind,
            acquired_at_unix_ms: value.acquired_at_unix_ms,
            footprint: value.footprint,
            modality: value.modality,
            bands: value.bands,
            uncertainty: value.uncertainty,
            source_digest: value.source_digest,
            processing_lineage: value.processing_lineage,
            metadata_digest: value.metadata_digest,
        };
        product.verify_digest()?;
        Ok(product)
    }
}

impl<'de> Deserialize<'de> for FrozenSentinelProduct {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = FrozenSentinelProductRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "source_kind", rename_all = "kebab-case")]
pub enum FixtureSourceRef {
    Product { product_id: String },
    Artifact { artifact_id: String },
}

impl FixtureSourceRef {
    fn label(&self) -> &str {
        match self {
            Self::Product { product_id } => product_id,
            Self::Artifact { artifact_id } => artifact_id,
        }
    }

    fn validate(&self) -> FixtureResult<()> {
        match self {
            Self::Product { product_id } => non_empty(product_id, "source product id"),
            Self::Artifact { artifact_id } => non_empty(artifact_id, "source artifact id"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FixtureArtifactKind {
    CalibratedRaster,
    MaskedRaster,
    TerrainCorrectedRaster,
    ResampledWindow,
    FeatureCube,
    Preview,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SentinelFixtureArtifact {
    pub artifact_id: String,
    pub kind: FixtureArtifactKind,
    pub content_digest: FrozenDigest,
    pub byte_len: Option<u64>,
    /// Ordered semantic inputs. Order remains identity-significant because band stacking and similar
    /// transforms may be order-sensitive.
    pub sources: Vec<FixtureSourceRef>,
    /// Ordered processing steps. Their order is identity-significant.
    pub processing_steps: Vec<FrozenProcessingStep>,
    pub identity_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct SentinelFixtureArtifactRepr {
    artifact_id: String,
    kind: FixtureArtifactKind,
    content_digest: FrozenDigest,
    byte_len: Option<u64>,
    sources: Vec<FixtureSourceRef>,
    processing_steps: Vec<FrozenProcessingStep>,
    identity_digest: String,
}

#[derive(Serialize)]
struct ArtifactDigestView<'a> {
    schema: &'static str,
    artifact_id: &'a str,
    kind: &'a FixtureArtifactKind,
    content_digest: &'a FrozenDigest,
    byte_len: Option<u64>,
    sources: &'a [FixtureSourceRef],
    processing_steps: &'a [FrozenProcessingStep],
}

impl SentinelFixtureArtifact {
    pub fn new(
        artifact_id: impl Into<String>,
        kind: FixtureArtifactKind,
        content_digest: FrozenDigest,
        byte_len: Option<u64>,
        sources: Vec<FixtureSourceRef>,
        processing_steps: Vec<FrozenProcessingStep>,
    ) -> FixtureResult<Self> {
        let mut artifact = Self {
            artifact_id: artifact_id.into(),
            kind,
            content_digest,
            byte_len,
            sources,
            processing_steps,
            identity_digest: String::new(),
        };
        artifact.validate_payload()?;
        artifact.identity_digest = artifact.compute_digest()?;
        Ok(artifact)
    }

    fn digest_view(&self) -> ArtifactDigestView<'_> {
        ArtifactDigestView {
            schema: ARTIFACT_SCHEMA,
            artifact_id: &self.artifact_id,
            kind: &self.kind,
            content_digest: &self.content_digest,
            byte_len: self.byte_len,
            sources: &self.sources,
            processing_steps: &self.processing_steps,
        }
    }

    pub fn compute_digest(&self) -> FixtureResult<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> FixtureResult<()> {
        self.validate_payload()?;
        validate_hex(&self.identity_digest, "artifact identity digest")?;
        if self.compute_digest()? != self.identity_digest {
            return Err(FixtureError::ArtifactDigestMismatch(self.artifact_id.clone()));
        }
        Ok(())
    }

    fn validate_payload(&self) -> FixtureResult<()> {
        non_empty(&self.artifact_id, "artifact id")?;
        self.content_digest.validate()?;
        if self.sources.is_empty() {
            return Err(FixtureError::MissingArtifactSource(self.artifact_id.clone()));
        }
        if self.processing_steps.is_empty() {
            return Err(FixtureError::MissingProcessingStep(self.artifact_id.clone()));
        }
        let mut seen = HashSet::new();
        for source in &self.sources {
            source.validate()?;
            if !seen.insert(source.clone()) {
                return Err(FixtureError::DuplicateSource {
                    artifact_id: self.artifact_id.clone(),
                    source_id: source.label().to_string(),
                });
            }
        }
        for step in &self.processing_steps {
            step.validate()?;
        }
        Ok(())
    }
}

impl TryFrom<SentinelFixtureArtifactRepr> for SentinelFixtureArtifact {
    type Error = FixtureError;

    fn try_from(value: SentinelFixtureArtifactRepr) -> FixtureResult<Self> {
        let artifact = Self {
            artifact_id: value.artifact_id,
            kind: value.kind,
            content_digest: value.content_digest,
            byte_len: value.byte_len,
            sources: value.sources,
            processing_steps: value.processing_steps,
            identity_digest: value.identity_digest,
        };
        artifact.verify_digest()?;
        Ok(artifact)
    }
}

impl<'de> Deserialize<'de> for SentinelFixtureArtifact {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = SentinelFixtureArtifactRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenSentinelFixtureManifest {
    pub fixture_id: String,
    pub products: Vec<FrozenSentinelProduct>,
    pub artifacts: Vec<SentinelFixtureArtifact>,
    pub manifest_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct FrozenSentinelFixtureManifestRepr {
    fixture_id: String,
    products: Vec<FrozenSentinelProduct>,
    artifacts: Vec<SentinelFixtureArtifact>,
    manifest_digest: String,
}

#[derive(Serialize)]
struct FixtureDigestView<'a> {
    schema: &'static str,
    fixture_id: &'a str,
    products: &'a [FrozenSentinelProduct],
    artifacts: &'a [SentinelFixtureArtifact],
}

impl FrozenSentinelFixtureManifest {
    pub fn new(
        fixture_id: impl Into<String>,
        products: Vec<SentinelProductMetadata>,
        mut artifacts: Vec<SentinelFixtureArtifact>,
    ) -> FixtureResult<Self> {
        let mut products = products
            .iter()
            .map(FrozenSentinelProduct::from_metadata)
            .collect::<FixtureResult<Vec<_>>>()?;
        products.sort_by(|a, b| a.product_id.cmp(&b.product_id));
        artifacts.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));

        let mut manifest = Self {
            fixture_id: fixture_id.into(),
            products,
            artifacts,
            manifest_digest: String::new(),
        };
        manifest.validate_payload()?;
        manifest.manifest_digest = manifest.compute_digest()?;
        Ok(manifest)
    }

    fn digest_view(&self) -> FixtureDigestView<'_> {
        FixtureDigestView {
            schema: FIXTURE_SCHEMA,
            fixture_id: &self.fixture_id,
            products: &self.products,
            artifacts: &self.artifacts,
        }
    }

    pub fn compute_digest(&self) -> FixtureResult<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> FixtureResult<()> {
        self.validate_payload()?;
        validate_hex(&self.manifest_digest, "fixture manifest digest")?;
        if self.compute_digest()? != self.manifest_digest {
            return Err(FixtureError::ManifestDigestMismatch);
        }
        Ok(())
    }

    fn validate_payload(&self) -> FixtureResult<()> {
        non_empty(&self.fixture_id, "fixture id")?;

        let mut product_ids = HashSet::new();
        for product in &self.products {
            product.verify_digest()?;
            if !product_ids.insert(product.product_id.clone()) {
                return Err(FixtureError::DuplicateProduct(product.product_id.clone()));
            }
        }

        let mut artifact_ids = HashSet::new();
        for artifact in &self.artifacts {
            artifact.verify_digest()?;
            if !artifact_ids.insert(artifact.artifact_id.clone()) {
                return Err(FixtureError::DuplicateArtifact(artifact.artifact_id.clone()));
            }
            if product_ids.contains(&artifact.artifact_id) {
                return Err(FixtureError::IdentifierCollision(artifact.artifact_id.clone()));
            }
        }

        for artifact in &self.artifacts {
            for source in &artifact.sources {
                match source {
                    FixtureSourceRef::Product { product_id } if !product_ids.contains(product_id) => {
                        return Err(FixtureError::UnknownProductSource {
                            artifact_id: artifact.artifact_id.clone(),
                            product_id: product_id.clone(),
                        });
                    }
                    FixtureSourceRef::Artifact { artifact_id }
                        if !artifact_ids.contains(artifact_id) =>
                    {
                        return Err(FixtureError::UnknownArtifactSource {
                            artifact_id: artifact.artifact_id.clone(),
                            source_artifact_id: artifact_id.clone(),
                        });
                    }
                    _ => {}
                }
            }
        }

        validate_acyclic_artifacts(&self.artifacts)
    }
}

impl TryFrom<FrozenSentinelFixtureManifestRepr> for FrozenSentinelFixtureManifest {
    type Error = FixtureError;

    fn try_from(value: FrozenSentinelFixtureManifestRepr) -> FixtureResult<Self> {
        let manifest = Self {
            fixture_id: value.fixture_id,
            products: value.products,
            artifacts: value.artifacts,
            manifest_digest: value.manifest_digest,
        };
        manifest.verify_digest()?;
        Ok(manifest)
    }
}

impl<'de> Deserialize<'de> for FrozenSentinelFixtureManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = FrozenSentinelFixtureManifestRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

fn validate_acyclic_artifacts(artifacts: &[SentinelFixtureArtifact]) -> FixtureResult<()> {
    let by_id: BTreeMap<String, &SentinelFixtureArtifact> = artifacts
        .iter()
        .map(|artifact| (artifact.artifact_id.clone(), artifact))
        .collect();
    let mut visiting = HashSet::<String>::new();
    let mut visited = HashSet::<String>::new();

    fn visit(
        id: &str,
        by_id: &BTreeMap<String, &SentinelFixtureArtifact>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> FixtureResult<()> {
        if visited.contains(id) {
            return Ok(());
        }
        if !visiting.insert(id.to_string()) {
            return Err(FixtureError::ArtifactCycle(id.to_string()));
        }
        if let Some(artifact) = by_id.get(id) {
            for source in &artifact.sources {
                if let FixtureSourceRef::Artifact { artifact_id } = source {
                    visit(artifact_id, by_id, visiting, visited)?;
                }
            }
        }
        visiting.remove(id);
        visited.insert(id.to_string());
        Ok(())
    }

    for id in by_id.keys() {
        visit(id, &by_id, &mut visiting, &mut visited)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_earth_observation::{
        BandDescriptor, Confidence, GeoFootprint, GeoPoint, ObservationUncertainty,
        ProcessingLineage,
    };

    fn product(id: &str, kind: SentinelProductKind, at: i64) -> SentinelProductMetadata {
        let (mission_id, instrument_id, modality) = match kind.mission() {
            SentinelMission::Sentinel1 => (
                "Sentinel-1",
                "C-SAR",
                SensorModality::SyntheticApertureRadar {
                    band: RadarBand::C,
                    polarization: Some(Polarization::Vv),
                },
            ),
            SentinelMission::Sentinel2 => ("Sentinel-2", "MSI", SensorModality::Multispectral),
        };
        let source_hex = if id.ends_with('1') {
            "11".repeat(32)
        } else {
            "22".repeat(32)
        };
        SentinelProductMetadata {
            observation_id: format!("obs-{id}"),
            mission_id: mission_id.into(),
            instrument_id: instrument_id.into(),
            product_id: id.into(),
            product_kind: kind,
            acquired_at_unix_ms: at,
            footprint: GeoFootprint::new(vec![
                GeoPoint::new(-25.0, 28.0).unwrap(),
                GeoPoint::new(-25.1, 28.0).unwrap(),
                GeoPoint::new(-25.0, 28.1).unwrap(),
            ])
            .unwrap(),
            modality,
            bands: vec![BandDescriptor::new(
                "fixture-band",
                Some(842.0),
                RadiometricUnit::Reflectance,
            )
            .unwrap()],
            uncertainty: ObservationUncertainty::new(
                Some(Confidence::new(0.9).unwrap()),
                Some(0.01),
                Some("fixture uncertainty".into()),
            )
            .unwrap(),
            source_digest: ContentDigest::new(DigestAlgorithm::Sha256, source_hex).unwrap(),
            lineage: ProcessingLineage::default(),
        }
    }

    fn step(name: &str) -> FrozenProcessingStep {
        FrozenProcessingStep {
            name: name.into(),
            software: "fixture-tool".into(),
            version: "1".into(),
            parameters_digest: Some(FrozenDigest {
                algorithm: "sha256".into(),
                hex: "abcd".into(),
            }),
        }
    }

    fn artifact(id: &str, sources: Vec<FixtureSourceRef>) -> SentinelFixtureArtifact {
        SentinelFixtureArtifact::new(
            id,
            FixtureArtifactKind::FeatureCube,
            FrozenDigest {
                algorithm: "sha256".into(),
                hex: format!("{:064x}", id.len()),
            },
            Some(1024),
            sources,
            vec![step("derive-feature-cube")],
        )
        .unwrap()
    }

    #[test]
    fn freezes_s1_s2_products_and_derived_lineage() {
        let manifest = FrozenSentinelFixtureManifest::new(
            "wetland-fixture-v1",
            vec![
                product("s2", SentinelProductKind::Sentinel2L2A, 200),
                product("s1", SentinelProductKind::Sentinel1Grd, 100),
            ],
            vec![artifact(
                "feature-cube",
                vec![
                    FixtureSourceRef::Product {
                        product_id: "s1".into(),
                    },
                    FixtureSourceRef::Product {
                        product_id: "s2".into(),
                    },
                ],
            )],
        )
        .unwrap();

        manifest.verify_digest().unwrap();
        assert_eq!(manifest.products[0].product_id, "s1");
        assert_eq!(manifest.products[1].product_id, "s2");
        assert_eq!(manifest.artifacts[0].artifact_id, "feature-cube");
    }

    #[test]
    fn manifest_order_is_canonical_for_product_and_artifact_lists() {
        let a = artifact(
            "a",
            vec![FixtureSourceRef::Product {
                product_id: "s1".into(),
            }],
        );
        let b = artifact(
            "b",
            vec![FixtureSourceRef::Artifact {
                artifact_id: "a".into(),
            }],
        );
        let first = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![
                product("s2", SentinelProductKind::Sentinel2L2A, 2),
                product("s1", SentinelProductKind::Sentinel1Grd, 1),
            ],
            vec![b.clone(), a.clone()],
        )
        .unwrap();
        let second = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![
                product("s1", SentinelProductKind::Sentinel1Grd, 1),
                product("s2", SentinelProductKind::Sentinel2L2A, 2),
            ],
            vec![a, b],
        )
        .unwrap();
        assert_eq!(first.manifest_digest, second.manifest_digest);
    }

    #[test]
    fn missing_source_reference_fails_closed() {
        let err = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![product("s1", SentinelProductKind::Sentinel1Grd, 1)],
            vec![artifact(
                "derived",
                vec![FixtureSourceRef::Product {
                    product_id: "missing".into(),
                }],
            )],
        )
        .unwrap_err();
        assert!(matches!(err, FixtureError::UnknownProductSource { .. }));
    }

    #[test]
    fn artifact_cycles_fail_even_when_inner_digests_are_recomputed() {
        let a = artifact(
            "a",
            vec![FixtureSourceRef::Artifact {
                artifact_id: "b".into(),
            }],
        );
        let b = artifact(
            "b",
            vec![FixtureSourceRef::Artifact {
                artifact_id: "a".into(),
            }],
        );
        let err = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![product("s1", SentinelProductKind::Sentinel1Grd, 1)],
            vec![a, b],
        )
        .unwrap_err();
        assert!(matches!(err, FixtureError::ArtifactCycle(_)));
    }

    #[test]
    fn persisted_manifest_rejects_tampering() {
        let manifest = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![product("s1", SentinelProductKind::Sentinel1Grd, 1)],
            vec![artifact(
                "derived",
                vec![FixtureSourceRef::Product {
                    product_id: "s1".into(),
                }],
            )],
        )
        .unwrap();
        let mut value = serde_json::to_value(&manifest).unwrap();
        value["artifacts"][0]["byte_len"] = serde_json::Value::from(999_u64);
        assert!(serde_json::from_value::<FrozenSentinelFixtureManifest>(value).is_err());
    }

    #[test]
    fn persisted_product_rejects_recomputed_outer_manifest_tamper() {
        let manifest = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![product("s1", SentinelProductKind::Sentinel1Grd, 1)],
            vec![],
        )
        .unwrap();
        let mut value = serde_json::to_value(&manifest).unwrap();
        value["products"][0]["acquired_at_unix_ms"] = serde_json::Value::from(999_i64);
        // Even if an attacker later recomputed the outer manifest digest, the nested product's own
        // metadata digest must first verify during deserialization.
        assert!(serde_json::from_value::<FrozenSentinelFixtureManifest>(value).is_err());
    }

    #[test]
    fn persisted_manifest_rejects_noncanonical_nested_digest_with_recomputed_identities() {
        let mut manifest = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![product("s1", SentinelProductKind::Sentinel1Grd, 1)],
            vec![artifact(
                "derived",
                vec![FixtureSourceRef::Product {
                    product_id: "s1".into(),
                }],
            )],
        )
        .unwrap();

        // This is the stronger canonicalization attack: mutate only the spelling of a nested
        // cryptographic value, then recompute every enclosing identity so no stale hash remains.
        // The scientific object must still fail persistence validation because v1 has exactly one
        // canonical textual representation for a digest.
        manifest.products[0].source_digest.hex = "AB".repeat(32);
        manifest.products[0].metadata_digest = manifest.products[0].compute_digest().unwrap();
        manifest.manifest_digest = manifest.compute_digest().unwrap();

        let value = serde_json::to_value(&manifest).unwrap();
        assert!(serde_json::from_value::<FrozenSentinelFixtureManifest>(value).is_err());
    }

    #[test]
    fn persisted_manifest_rejects_noncanonical_algorithm_with_recomputed_identities() {
        let mut manifest = FrozenSentinelFixtureManifest::new(
            "fixture",
            vec![product("s1", SentinelProductKind::Sentinel1Grd, 1)],
            vec![],
        )
        .unwrap();

        manifest.products[0].source_digest.algorithm = "SHA256".into();
        manifest.products[0].metadata_digest = manifest.products[0].compute_digest().unwrap();
        manifest.manifest_digest = manifest.compute_digest().unwrap();

        let value = serde_json::to_value(&manifest).unwrap();
        assert!(serde_json::from_value::<FrozenSentinelFixtureManifest>(value).is_err());
    }

    #[test]
    fn source_order_is_semantic_for_derived_artifacts() {
        let first = artifact(
            "stack",
            vec![
                FixtureSourceRef::Product {
                    product_id: "s1".into(),
                },
                FixtureSourceRef::Product {
                    product_id: "s2".into(),
                },
            ],
        );
        let second = artifact(
            "stack",
            vec![
                FixtureSourceRef::Product {
                    product_id: "s2".into(),
                },
                FixtureSourceRef::Product {
                    product_id: "s1".into(),
                },
            ],
        );
        assert_ne!(first.identity_digest, second.identity_digest);
    }

    #[test]
    fn product_metadata_digest_changes_when_exact_float_bits_change() {
        let first = product("s2", SentinelProductKind::Sentinel2L2A, 1);
        let mut second = first.clone();
        second.footprint.vertices[0].latitude_deg = -25.000000000000004;
        let first = FrozenSentinelProduct::from_metadata(&first).unwrap();
        let second = FrozenSentinelProduct::from_metadata(&second).unwrap();
        assert_ne!(first.metadata_digest, second.metadata_digest);
    }
}
