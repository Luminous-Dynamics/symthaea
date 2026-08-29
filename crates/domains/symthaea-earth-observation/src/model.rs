use std::error::Error;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, EvidenceError>;

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceError {
    EmptyId(&'static str),
    InvalidLatitude(f64),
    InvalidLongitude(f64),
    NonFinite(&'static str),
    Negative(&'static str, f64),
    InvalidConfidence(f64),
    InvalidFootprintVertexCount(usize),
    InvalidDigest,
    MissingSupport,
    MissingReferencedObservation(String),
    DirectSubsurfaceClaimUnsupported,
    DirectSubsurfaceDepthUnsupported {
        requested_depth_m: f64,
        validated_depth_m: f64,
    },
}

impl Display for EvidenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId(kind) => write!(f, "{kind} must not be empty"),
            Self::InvalidLatitude(value) => write!(f, "latitude {value} is outside [-90, 90]"),
            Self::InvalidLongitude(value) => {
                write!(f, "longitude {value} is outside [-180, 180]")
            }
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::Negative(field, value) => write!(f, "{field} must be non-negative, got {value}"),
            Self::InvalidConfidence(value) => {
                write!(f, "confidence must be in [0, 1], got {value}")
            }
            Self::InvalidFootprintVertexCount(count) => {
                write!(f, "a footprint polygon requires at least 3 vertices, got {count}")
            }
            Self::InvalidDigest => write!(f, "content digest must be non-empty hexadecimal text"),
            Self::MissingSupport => write!(f, "evidence-bearing claims require support"),
            Self::MissingReferencedObservation(id) => {
                write!(f, "support references unknown observation {id}")
            }
            Self::DirectSubsurfaceClaimUnsupported => write!(
                f,
                "direct subsurface claims require directly penetrating calibrated evidence"
            ),
            Self::DirectSubsurfaceDepthUnsupported {
                requested_depth_m,
                validated_depth_m,
            } => write!(
                f,
                "requested direct subsurface depth {requested_depth_m} m exceeds validated sensor depth {validated_depth_m} m"
            ),
        }
    }
}

impl Error for EvidenceError {}

fn require_id(value: impl Into<String>, kind: &'static str) -> Result<String> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(EvidenceError::EmptyId(kind));
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f64);

impl Confidence {
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(EvidenceError::InvalidConfidence(value));
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
}

impl GeoPoint {
    pub fn new(latitude_deg: f64, longitude_deg: f64) -> Result<Self> {
        if !latitude_deg.is_finite() {
            return Err(EvidenceError::NonFinite("latitude_deg"));
        }
        if !longitude_deg.is_finite() {
            return Err(EvidenceError::NonFinite("longitude_deg"));
        }
        if !(-90.0..=90.0).contains(&latitude_deg) {
            return Err(EvidenceError::InvalidLatitude(latitude_deg));
        }
        if !(-180.0..=180.0).contains(&longitude_deg) {
            return Err(EvidenceError::InvalidLongitude(longitude_deg));
        }
        Ok(Self {
            latitude_deg,
            longitude_deg,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeoFootprint {
    pub vertices: Vec<GeoPoint>,
}

impl GeoFootprint {
    pub fn new(vertices: Vec<GeoPoint>) -> Result<Self> {
        if vertices.len() < 3 {
            return Err(EvidenceError::InvalidFootprintVertexCount(vertices.len()));
        }
        Ok(Self { vertices })
    }
}

macro_rules! id_type {
    ($name:ident, $label:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self> {
                Ok(Self(require_id(value, $label)?))
            }
        }
    };
}

id_type!(MissionId, "mission id");
id_type!(InstrumentId, "instrument id");
id_type!(ProductId, "product id");
id_type!(ObservationId, "observation id");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestAlgorithm {
    Sha256,
    Blake3,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentDigest {
    pub algorithm: DigestAlgorithm,
    pub hex: String,
}

impl ContentDigest {
    pub fn new(algorithm: DigestAlgorithm, hex: impl Into<String>) -> Result<Self> {
        let hex = hex.into();
        if hex.is_empty() || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(EvidenceError::InvalidDigest);
        }
        Ok(Self { algorithm, hex })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadarBand {
    P,
    L,
    S,
    C,
    X,
    Ku,
    Ka,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarization {
    Hh,
    Hv,
    Vh,
    Vv,
    Circular,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SensorModality {
    Optical,
    Multispectral,
    Hyperspectral,
    SyntheticApertureRadar {
        band: RadarBand,
        polarization: Option<Polarization>,
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
    Other(String),
}

/// Acquisition-specific calibrated sensitivity. This intentionally contains no
/// universal wavelength-to-depth lookup: medium, moisture, geometry,
/// processing, and validation determine defensible penetration bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObservationSensitivity {
    SurfaceOnly,
    IndirectSubsurface,
    DirectPenetrating { max_validated_depth_m: f64 },
}

impl ObservationSensitivity {
    pub fn direct_penetrating(max_validated_depth_m: f64) -> Result<Self> {
        if !max_validated_depth_m.is_finite() {
            return Err(EvidenceError::NonFinite("max_validated_depth_m"));
        }
        if max_validated_depth_m < 0.0 {
            return Err(EvidenceError::Negative(
                "max_validated_depth_m",
                max_validated_depth_m,
            ));
        }
        Ok(Self::DirectPenetrating {
            max_validated_depth_m,
        })
    }

    pub fn supports_direct_depth(self, depth_m: Option<f64>) -> Result<bool> {
        let Some(depth_m) = depth_m else {
            return Ok(matches!(self, Self::DirectPenetrating { .. }));
        };
        if !depth_m.is_finite() {
            return Err(EvidenceError::NonFinite("estimated_depth_m"));
        }
        if depth_m < 0.0 {
            return Err(EvidenceError::Negative("estimated_depth_m", depth_m));
        }
        Ok(matches!(
            self,
            Self::DirectPenetrating {
                max_validated_depth_m
            } if depth_m <= max_validated_depth_m
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiometricUnit {
    Reflectance,
    BrightnessTemperature,
    BackscatterLinear,
    BackscatterDecibel,
    Radiance,
    Dimensionless,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BandDescriptor {
    pub name: String,
    pub center_wavelength_nm: Option<f64>,
    pub unit: RadiometricUnit,
}

impl BandDescriptor {
    pub fn new(
        name: impl Into<String>,
        center_wavelength_nm: Option<f64>,
        unit: RadiometricUnit,
    ) -> Result<Self> {
        if let Some(value) = center_wavelength_nm {
            if !value.is_finite() {
                return Err(EvidenceError::NonFinite("center_wavelength_nm"));
            }
            if value < 0.0 {
                return Err(EvidenceError::Negative("center_wavelength_nm", value));
            }
        }
        Ok(Self {
            name: require_id(name, "band name")?,
            center_wavelength_nm,
            unit,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationUncertainty {
    pub confidence: Option<Confidence>,
    pub standard_uncertainty: Option<f64>,
    pub note: Option<String>,
}

impl ObservationUncertainty {
    pub fn new(
        confidence: Option<Confidence>,
        standard_uncertainty: Option<f64>,
        note: Option<String>,
    ) -> Result<Self> {
        if let Some(value) = standard_uncertainty {
            if !value.is_finite() {
                return Err(EvidenceError::NonFinite("standard_uncertainty"));
            }
            if value < 0.0 {
                return Err(EvidenceError::Negative("standard_uncertainty", value));
            }
        }
        Ok(Self {
            confidence,
            standard_uncertainty,
            note,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingStep {
    pub name: String,
    pub software: String,
    pub version: String,
    pub parameters_digest: Option<ContentDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProcessingLineage {
    pub steps: Vec<ProcessingStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationEvidence {
    pub id: ObservationId,
    pub mission: MissionId,
    pub instrument: InstrumentId,
    pub product: ProductId,
    pub acquired_at_unix_ms: i64,
    pub footprint: GeoFootprint,
    pub modality: SensorModality,
    pub sensitivity: ObservationSensitivity,
    pub bands: Vec<BandDescriptor>,
    pub uncertainty: ObservationUncertainty,
    pub source_digest: ContentDigest,
    pub lineage: ProcessingLineage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStage {
    Observation,
    Measurement,
    DerivedFeature,
    Hypothesis,
    Inference,
    Prediction,
    Verification,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    pub id: String,
    pub stage: EvidenceStage,
}

impl EvidenceRef {
    pub fn new(id: impl Into<String>, stage: EvidenceStage) -> Result<Self> {
        Ok(Self {
            id: require_id(id, "evidence reference id")?,
            stage,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub id: String,
    pub quantity: String,
    pub value: f64,
    pub unit: String,
    pub uncertainty: Option<ObservationUncertainty>,
    pub support: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralIndex {
    Ndvi,
    McFeetersNdwi,
    GaoNdwi,
    Nbr,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DerivedFeature {
    pub id: String,
    pub name: String,
    pub value: Option<f64>,
    pub confidence: Option<Confidence>,
    pub support: Vec<EvidenceRef>,
}

impl DerivedFeature {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        value: Option<f64>,
        confidence: Option<Confidence>,
        support: Vec<EvidenceRef>,
    ) -> Result<Self> {
        if let Some(value) = value {
            if !value.is_finite() {
                return Err(EvidenceError::NonFinite("derived feature value"));
            }
        }
        if support.is_empty() {
            return Err(EvidenceError::MissingSupport);
        }
        Ok(Self {
            id: require_id(id, "derived feature id")?,
            name: require_id(name, "derived feature name")?,
            value,
            confidence,
            support,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HypothesisDomain {
    Surface,
    Subsurface { estimated_depth_m: Option<f64> },
    Atmosphere,
    Infrastructure,
    Biosphere,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimMode {
    DirectObservation,
    IndirectInference,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub id: String,
    pub statement: String,
    pub domain: HypothesisDomain,
    pub mode: ClaimMode,
    pub confidence: Confidence,
    pub supporting_observations: Vec<ObservationId>,
}

impl Hypothesis {
    pub fn new(
        id: impl Into<String>,
        statement: impl Into<String>,
        domain: HypothesisDomain,
        mode: ClaimMode,
        confidence: Confidence,
        supporting_observations: Vec<ObservationId>,
    ) -> Result<Self> {
        if supporting_observations.is_empty() {
            return Err(EvidenceError::MissingSupport);
        }
        Ok(Self {
            id: require_id(id, "hypothesis id")?,
            statement: require_id(statement, "hypothesis statement")?,
            domain,
            mode,
            confidence,
            supporting_observations,
        })
    }

    pub fn validate_observation_support(&self, observations: &[ObservationEvidence]) -> Result<()> {
        let mut matched = Vec::with_capacity(self.supporting_observations.len());
        for requested in &self.supporting_observations {
            let Some(observation) = observations.iter().find(|candidate| &candidate.id == requested)
            else {
                return Err(EvidenceError::MissingReferencedObservation(requested.0.clone()));
            };
            matched.push(observation);
        }

        let HypothesisDomain::Subsurface { estimated_depth_m } = &self.domain else {
            return Ok(());
        };
        let estimated_depth_m = *estimated_depth_m;

        if self.mode == ClaimMode::IndirectInference {
            return Ok(());
        }

        let mut deepest_validated = None::<f64>;
        for observation in matched {
            if let ObservationSensitivity::DirectPenetrating {
                max_validated_depth_m,
            } = observation.sensitivity
            {
                deepest_validated = Some(
                    deepest_validated
                        .map(|current| current.max(max_validated_depth_m))
                        .unwrap_or(max_validated_depth_m),
                );
                if observation.sensitivity.supports_direct_depth(estimated_depth_m)? {
                    return Ok(());
                }
            }
        }

        match (estimated_depth_m, deepest_validated) {
            (Some(requested_depth_m), Some(validated_depth_m)) => {
                Err(EvidenceError::DirectSubsurfaceDepthUnsupported {
                    requested_depth_m,
                    validated_depth_m,
                })
            }
            _ => Err(EvidenceError::DirectSubsurfaceClaimUnsupported),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inference {
    pub id: String,
    pub conclusion: String,
    pub confidence: Confidence,
    pub alternatives: Vec<String>,
    pub support: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub id: String,
    pub statement: String,
    pub valid_from_unix_ms: i64,
    pub valid_until_unix_ms: i64,
    pub confidence: Confidence,
    pub support: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Confirmed,
    Refuted,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Verification {
    pub id: String,
    pub target_id: String,
    pub status: VerificationStatus,
    pub confidence: Confidence,
    pub support: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceConflict {
    pub id: String,
    pub competing_evidence: Vec<EvidenceRef>,
    pub note: String,
}

impl EvidenceConflict {
    pub fn new(
        id: impl Into<String>,
        competing_evidence: Vec<EvidenceRef>,
        note: impl Into<String>,
    ) -> Result<Self> {
        if competing_evidence.len() < 2 {
            return Err(EvidenceError::MissingSupport);
        }
        Ok(Self {
            id: require_id(id, "evidence conflict id")?,
            competing_evidence,
            note: require_id(note, "evidence conflict note")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> GeoFootprint {
        GeoFootprint::new(vec![
            GeoPoint::new(-25.0, 28.0).unwrap(),
            GeoPoint::new(-25.1, 28.0).unwrap(),
            GeoPoint::new(-25.0, 28.1).unwrap(),
        ])
        .unwrap()
    }

    fn observation(id: &str, sensitivity: ObservationSensitivity) -> ObservationEvidence {
        ObservationEvidence {
            id: ObservationId::new(id).unwrap(),
            mission: MissionId::new("fixture-mission").unwrap(),
            instrument: InstrumentId::new("fixture-instrument").unwrap(),
            product: ProductId::new("fixture-product").unwrap(),
            acquired_at_unix_ms: 0,
            footprint: triangle(),
            modality: SensorModality::SyntheticApertureRadar {
                band: RadarBand::L,
                polarization: Some(Polarization::Vv),
            },
            sensitivity,
            bands: vec![],
            uncertainty: ObservationUncertainty::new(
                Some(Confidence::new(0.9).unwrap()),
                None,
                None,
            )
            .unwrap(),
            source_digest: ContentDigest::new(DigestAlgorithm::Sha256, "00").unwrap(),
            lineage: ProcessingLineage::default(),
        }
    }

    #[test]
    fn confidence_rejects_out_of_range_values() {
        assert!(Confidence::new(-0.01).is_err());
        assert!(Confidence::new(1.01).is_err());
        assert!(Confidence::new(0.5).is_ok());
    }

    #[test]
    fn direct_subsurface_claim_rejects_indirect_evidence() {
        let observations = vec![observation(
            "obs-1",
            ObservationSensitivity::IndirectSubsurface,
        )];
        let hypothesis = Hypothesis::new(
            "hyp-1",
            "candidate buried structure",
            HypothesisDomain::Subsurface {
                estimated_depth_m: Some(1.0),
            },
            ClaimMode::DirectObservation,
            Confidence::new(0.7).unwrap(),
            vec![ObservationId::new("obs-1").unwrap()],
        )
        .unwrap();

        assert_eq!(
            hypothesis.validate_observation_support(&observations),
            Err(EvidenceError::DirectSubsurfaceClaimUnsupported)
        );
    }

    #[test]
    fn indirect_subsurface_hypothesis_can_use_indirect_evidence() {
        let observations = vec![observation(
            "obs-1",
            ObservationSensitivity::IndirectSubsurface,
        )];
        let hypothesis = Hypothesis::new(
            "hyp-1",
            "surface deformation is consistent with a subsurface process",
            HypothesisDomain::Subsurface {
                estimated_depth_m: None,
            },
            ClaimMode::IndirectInference,
            Confidence::new(0.6).unwrap(),
            vec![ObservationId::new("obs-1").unwrap()],
        )
        .unwrap();

        assert!(hypothesis.validate_observation_support(&observations).is_ok());
    }

    #[test]
    fn direct_subsurface_claim_respects_validated_depth() {
        let observations = vec![observation(
            "obs-1",
            ObservationSensitivity::direct_penetrating(2.0).unwrap(),
        )];

        let supported = Hypothesis::new(
            "hyp-near",
            "directly observed shallow reflector",
            HypothesisDomain::Subsurface {
                estimated_depth_m: Some(1.5),
            },
            ClaimMode::DirectObservation,
            Confidence::new(0.8).unwrap(),
            vec![ObservationId::new("obs-1").unwrap()],
        )
        .unwrap();
        assert!(supported.validate_observation_support(&observations).is_ok());

        let too_deep = Hypothesis::new(
            "hyp-deep",
            "directly observed deep reflector",
            HypothesisDomain::Subsurface {
                estimated_depth_m: Some(3.0),
            },
            ClaimMode::DirectObservation,
            Confidence::new(0.8).unwrap(),
            vec![ObservationId::new("obs-1").unwrap()],
        )
        .unwrap();
        assert_eq!(
            too_deep.validate_observation_support(&observations),
            Err(EvidenceError::DirectSubsurfaceDepthUnsupported {
                requested_depth_m: 3.0,
                validated_depth_m: 2.0,
            })
        );
    }

    #[test]
    fn named_ndwi_variants_remain_distinct() {
        assert_ne!(SpectralIndex::McFeetersNdwi, SpectralIndex::GaoNdwi);
    }
}
