//! Canonical raster evidence join for Planetary Perception.
//!
//! This module composes two already-separate responsibilities:
//! - exact-window provenance answers which frozen source/pixels produced bytes;
//! - raster-payload semantics answer how those bytes are interpreted as samples.
//!
//! The join does not decode files, extract features, resample, or infer metadata.

use serde::{Deserialize, Serialize};
use symthaea_earth_observation::{
    payload::{
        BandInterleave, NoDataValue, RasterByteOrder, RasterPayloadDescriptor,
        RasterSampleType, ValidityMaskSemantics,
    },
    DigestAlgorithm,
};
use thiserror::Error;

use crate::{ExactPixelWindowEvidence, FrozenSentinelFixtureManifest};

const PAYLOAD_SEMANTICS_SCHEMA: &str = "symthaea-canonical-raster-payload-semantics/v1";
const CANONICAL_RASTER_EVIDENCE_SCHEMA: &str = "symthaea-canonical-raster-evidence/v1";
const HEX_256_LEN: usize = 64;

pub type CanonicalRasterResult<T> = std::result::Result<T, CanonicalRasterError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CanonicalRasterError {
    #[error("exact-window verification failed: {0}")]
    Window(String),
    #[error("payload digest is not canonical: {0}")]
    NonCanonicalPayloadDigest(String),
    #[error("payload digest does not match exact-window output bytes")]
    ContentDigestMismatch,
    #[error("payload shape {payload_rows}x{payload_cols} does not match exact-window output {window_rows}x{window_cols}")]
    ShapeMismatch {
        payload_rows: u32,
        payload_cols: u32,
        window_rows: u32,
        window_cols: u32,
    },
    #[error("payload byte length {payload} does not match exact-window byte length {window}")]
    ByteLengthMismatch { payload: u64, window: u64 },
    #[error("canonical raster receipt digest does not match recomputation")]
    ReceiptDigestMismatch,
    #[error("canonical raster evidence does not match authoritative inputs")]
    AuthoritativeMismatch,
    #[error("canonical raster serialization failed: {0}")]
    Serialization(String),
}

fn blake3_json<T: Serialize>(value: &T) -> CanonicalRasterResult<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CanonicalRasterError::Serialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn canonical_hex(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !value.bytes().any(|byte| byte.is_ascii_uppercase())
}

fn digest_algorithm_tag(algorithm: &DigestAlgorithm) -> &'static str {
    match algorithm {
        DigestAlgorithm::Sha256 => "sha256",
        DigestAlgorithm::Blake3 => "blake3",
        DigestAlgorithm::Other => "other",
    }
}

fn canonical_payload_digest(
    payload: &RasterPayloadDescriptor,
) -> CanonicalRasterResult<(&'static str, String)> {
    let digest = payload.content_digest();
    let algorithm = digest_algorithm_tag(&digest.algorithm);
    if !canonical_hex(&digest.hex) {
        return Err(CanonicalRasterError::NonCanonicalPayloadDigest(
            "hex must be lowercase hexadecimal".to_string(),
        ));
    }
    if matches!(algorithm, "sha256" | "blake3") && digest.hex.len() != HEX_256_LEN {
        return Err(CanonicalRasterError::NonCanonicalPayloadDigest(format!(
            "{algorithm} must contain exactly {HEX_256_LEN} hexadecimal characters"
        )));
    }
    Ok((algorithm, digest.hex.clone()))
}

fn sample_type_tag(value: RasterSampleType) -> &'static str {
    match value {
        RasterSampleType::U8 => "u8",
        RasterSampleType::I8 => "i8",
        RasterSampleType::U16 => "u16",
        RasterSampleType::I16 => "i16",
        RasterSampleType::U32 => "u32",
        RasterSampleType::I32 => "i32",
        RasterSampleType::F32 => "f32",
        RasterSampleType::F64 => "f64",
    }
}

fn byte_order_tag(value: RasterByteOrder) -> &'static str {
    match value {
        RasterByteOrder::NotApplicable => "not-applicable",
        RasterByteOrder::LittleEndian => "little-endian",
        RasterByteOrder::BigEndian => "big-endian",
    }
}

fn interleave_tag(value: BandInterleave) -> &'static str {
    match value {
        BandInterleave::BandSequential => "bsq",
        BandInterleave::BandInterleavedByLine => "bil",
        BandInterleave::BandInterleavedByPixel => "bip",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct NoDataIdentity {
    kind: &'static str,
    unsigned: Option<u64>,
    signed: Option<i64>,
    bits: Option<u64>,
}

fn nodata_identity(value: NoDataValue) -> NoDataIdentity {
    match value {
        NoDataValue::Unsigned(value) => NoDataIdentity {
            kind: "unsigned",
            unsigned: Some(value),
            signed: None,
            bits: None,
        },
        NoDataValue::Signed(value) => NoDataIdentity {
            kind: "signed",
            unsigned: None,
            signed: Some(value),
            bits: None,
        },
        NoDataValue::Float32Bits(value) => NoDataIdentity {
            kind: "f32-bits",
            unsigned: None,
            signed: None,
            bits: Some(value as u64),
        },
        NoDataValue::Float64Bits(value) => NoDataIdentity {
            kind: "f64-bits",
            unsigned: None,
            signed: None,
            bits: Some(value),
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct BandIdentity {
    name: String,
    scale_bits: u64,
    offset_bits: u64,
    nodata: Option<NoDataIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct MaskIdentity {
    kind: &'static str,
    band_index: Option<u16>,
    mask_id: Option<String>,
    valid_when_nonzero: Option<bool>,
}

fn mask_identity(value: &ValidityMaskSemantics) -> MaskIdentity {
    match value {
        ValidityMaskSemantics::None => MaskIdentity {
            kind: "none",
            band_index: None,
            mask_id: None,
            valid_when_nonzero: None,
        },
        ValidityMaskSemantics::EmbeddedBand {
            band_index,
            valid_when_nonzero,
        } => MaskIdentity {
            kind: "embedded-band",
            band_index: Some(*band_index),
            mask_id: None,
            valid_when_nonzero: Some(*valid_when_nonzero),
        },
        ValidityMaskSemantics::External {
            mask_id,
            valid_when_nonzero,
        } => MaskIdentity {
            kind: "external",
            band_index: None,
            mask_id: Some(mask_id.clone()),
            valid_when_nonzero: Some(*valid_when_nonzero),
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PayloadSemanticsIdentity {
    schema: &'static str,
    rows: u32,
    cols: u32,
    sample_type: &'static str,
    byte_order: &'static str,
    interleave: &'static str,
    bands: Vec<BandIdentity>,
    mask: MaskIdentity,
    byte_len: u64,
}

fn payload_semantics_digest(payload: &RasterPayloadDescriptor) -> CanonicalRasterResult<String> {
    let bands = payload
        .bands()
        .iter()
        .map(|band| {
            let transform = band.transform();
            BandIdentity {
                name: band.name().to_string(),
                scale_bits: transform.scale.to_bits(),
                offset_bits: transform.offset.to_bits(),
                nodata: band.nodata().map(nodata_identity),
            }
        })
        .collect();
    let shape = payload.shape();
    blake3_json(&PayloadSemanticsIdentity {
        schema: PAYLOAD_SEMANTICS_SCHEMA,
        rows: shape.rows(),
        cols: shape.cols(),
        sample_type: sample_type_tag(payload.sample_type()),
        byte_order: byte_order_tag(payload.byte_order()),
        interleave: interleave_tag(payload.interleave()),
        bands,
        mask: mask_identity(payload.validity_mask()),
        byte_len: payload.byte_len(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalRasterEvidence {
    exact_window_evidence_digest: String,
    fixture_manifest_digest: String,
    source_identity_digest: String,
    payload_semantics_digest: String,
    content_algorithm: String,
    content_hex: String,
    rows: u32,
    cols: u32,
    byte_len: u64,
    receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct CanonicalRasterEvidenceRepr {
    exact_window_evidence_digest: String,
    fixture_manifest_digest: String,
    source_identity_digest: String,
    payload_semantics_digest: String,
    content_algorithm: String,
    content_hex: String,
    rows: u32,
    cols: u32,
    byte_len: u64,
    receipt_digest: String,
}

#[derive(Serialize)]
struct ReceiptDigestView<'a> {
    schema: &'static str,
    exact_window_evidence_digest: &'a str,
    fixture_manifest_digest: &'a str,
    source_identity_digest: &'a str,
    payload_semantics_digest: &'a str,
    content_algorithm: &'a str,
    content_hex: &'a str,
    rows: u32,
    cols: u32,
    byte_len: u64,
}

impl CanonicalRasterEvidence {
    pub fn new_for_fixture(
        fixture: &FrozenSentinelFixtureManifest,
        window: &ExactPixelWindowEvidence,
        payload: &RasterPayloadDescriptor,
    ) -> CanonicalRasterResult<Self> {
        window
            .verify_against_fixture(fixture)
            .map_err(|error| CanonicalRasterError::Window(error.to_string()))?;

        let (payload_algorithm, payload_hex) = canonical_payload_digest(payload)?;
        if window.output_content_digest.algorithm != payload_algorithm
            || window.output_content_digest.hex != payload_hex
        {
            return Err(CanonicalRasterError::ContentDigestMismatch);
        }

        let shape = payload.shape();
        if shape.rows() != window.geometry.output_rows || shape.cols() != window.geometry.output_cols {
            return Err(CanonicalRasterError::ShapeMismatch {
                payload_rows: shape.rows(),
                payload_cols: shape.cols(),
                window_rows: window.geometry.output_rows,
                window_cols: window.geometry.output_cols,
            });
        }
        if let Some(window_byte_len) = window.byte_len {
            if window_byte_len != payload.byte_len() {
                return Err(CanonicalRasterError::ByteLengthMismatch {
                    payload: payload.byte_len(),
                    window: window_byte_len,
                });
            }
        }

        let mut receipt = Self {
            exact_window_evidence_digest: window.evidence_digest.clone(),
            fixture_manifest_digest: window.fixture_manifest_digest.clone(),
            source_identity_digest: window.source_identity_digest.clone(),
            payload_semantics_digest: payload_semantics_digest(payload)?,
            content_algorithm: payload_algorithm.to_string(),
            content_hex: payload_hex,
            rows: shape.rows(),
            cols: shape.cols(),
            byte_len: payload.byte_len(),
            receipt_digest: String::new(),
        };
        receipt.receipt_digest = receipt.compute_digest()?;
        Ok(receipt)
    }

    fn digest_view(&self) -> ReceiptDigestView<'_> {
        ReceiptDigestView {
            schema: CANONICAL_RASTER_EVIDENCE_SCHEMA,
            exact_window_evidence_digest: &self.exact_window_evidence_digest,
            fixture_manifest_digest: &self.fixture_manifest_digest,
            source_identity_digest: &self.source_identity_digest,
            payload_semantics_digest: &self.payload_semantics_digest,
            content_algorithm: &self.content_algorithm,
            content_hex: &self.content_hex,
            rows: self.rows,
            cols: self.cols,
            byte_len: self.byte_len,
        }
    }

    pub fn compute_digest(&self) -> CanonicalRasterResult<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> CanonicalRasterResult<()> {
        for (name, value) in [
            ("window evidence", self.exact_window_evidence_digest.as_str()),
            ("fixture manifest", self.fixture_manifest_digest.as_str()),
            ("source identity", self.source_identity_digest.as_str()),
            ("payload semantics", self.payload_semantics_digest.as_str()),
            ("receipt", self.receipt_digest.as_str()),
        ] {
            if !canonical_hex(value) || value.len() != HEX_256_LEN {
                return Err(CanonicalRasterError::NonCanonicalPayloadDigest(format!(
                    "{name} digest must be 64 lowercase hexadecimal characters"
                )));
            }
        }
        if !matches!(self.content_algorithm.as_str(), "sha256" | "blake3" | "other")
            || !canonical_hex(&self.content_hex)
            || (matches!(self.content_algorithm.as_str(), "sha256" | "blake3")
                && self.content_hex.len() != HEX_256_LEN)
        {
            return Err(CanonicalRasterError::NonCanonicalPayloadDigest(
                "receipt content digest is not canonical".to_string(),
            ));
        }
        if self.compute_digest()? != self.receipt_digest {
            return Err(CanonicalRasterError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against(
        &self,
        fixture: &FrozenSentinelFixtureManifest,
        window: &ExactPixelWindowEvidence,
        payload: &RasterPayloadDescriptor,
    ) -> CanonicalRasterResult<()> {
        self.verify_digest()?;
        let expected = Self::new_for_fixture(fixture, window, payload)?;
        if self != &expected {
            return Err(CanonicalRasterError::AuthoritativeMismatch);
        }
        Ok(())
    }

    pub fn receipt_digest(&self) -> &str {
        &self.receipt_digest
    }

    pub fn payload_semantics_digest(&self) -> &str {
        &self.payload_semantics_digest
    }
}

impl TryFrom<CanonicalRasterEvidenceRepr> for CanonicalRasterEvidence {
    type Error = CanonicalRasterError;

    fn try_from(value: CanonicalRasterEvidenceRepr) -> CanonicalRasterResult<Self> {
        let receipt = Self {
            exact_window_evidence_digest: value.exact_window_evidence_digest,
            fixture_manifest_digest: value.fixture_manifest_digest,
            source_identity_digest: value.source_identity_digest,
            payload_semantics_digest: value.payload_semantics_digest,
            content_algorithm: value.content_algorithm,
            content_hex: value.content_hex,
            rows: value.rows,
            cols: value.cols,
            byte_len: value.byte_len,
            receipt_digest: value.receipt_digest,
        };
        receipt.verify_digest()?;
        Ok(receipt)
    }
}

impl<'de> Deserialize<'de> for CanonicalRasterEvidence {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = CanonicalRasterEvidenceRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FixtureSourceRef, FrozenDigest, FrozenSentinelFixtureManifest, SentinelProductKind,
        SentinelProductMetadata,
    };
    use symthaea_earth_observation::{
        payload::{RasterBandSemantics, SampleTransform},
        AffineGridTransform, ContentDigest, CrsId, GeoFootprint, GeoPoint, GridAnchor,
        ObservationUncertainty, PixelWindow, ProcessingLineage, RasterGrid, RasterReference,
        RasterShape, SensorModality,
    };

    fn hex256(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn fixture() -> FrozenSentinelFixtureManifest {
        let product = SentinelProductMetadata {
            observation_id: "obs-S2-L2A-001".to_string(),
            mission_id: "Sentinel-2".to_string(),
            instrument_id: "MSI".to_string(),
            product_id: "S2-L2A-001".to_string(),
            product_kind: SentinelProductKind::Sentinel2L2A,
            acquired_at_unix_ms: 100,
            footprint: GeoFootprint::new(vec![
                GeoPoint::new(-25.0, 28.0).unwrap(),
                GeoPoint::new(-25.1, 28.0).unwrap(),
                GeoPoint::new(-25.0, 28.1).unwrap(),
            ])
            .unwrap(),
            modality: SensorModality::Multispectral,
            bands: vec![],
            uncertainty: ObservationUncertainty::new(None, None, None).unwrap(),
            source_digest: ContentDigest::new(DigestAlgorithm::Sha256, hex256('1')).unwrap(),
            lineage: ProcessingLineage::default(),
        };
        FrozenSentinelFixtureManifest::new("fixture", vec![product], vec![]).unwrap()
    }

    fn plan() -> symthaea_earth_observation::RasterWindowPlan {
        let reference = RasterReference::new(
            CrsId::new("EPSG:32635").unwrap(),
            AffineGridTransform::new(
                500_000.0,
                7_200_000.0,
                10.0,
                0.0,
                0.0,
                -10.0,
                GridAnchor::PixelCorner,
            )
            .unwrap(),
        );
        RasterGrid::new(RasterShape::new(4, 5).unwrap(), reference)
            .window(PixelWindow::new(1, 1, 2, 3).unwrap())
            .unwrap()
    }

    fn window(fixture: &FrozenSentinelFixtureManifest) -> ExactPixelWindowEvidence {
        ExactPixelWindowEvidence::new_for_fixture(
            fixture,
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(),
            FrozenDigest {
                algorithm: "sha256".to_string(),
                hex: hex256('2'),
            },
            Some(24),
            "fixture-extractor",
            "1",
        )
        .unwrap()
    }

    fn payload(byte_order: RasterByteOrder, digest_char: char) -> RasterPayloadDescriptor {
        RasterPayloadDescriptor::new(
            RasterShape::new(2, 3).unwrap(),
            RasterSampleType::U16,
            byte_order,
            BandInterleave::BandInterleavedByPixel,
            vec![
                RasterBandSemantics::new("red", SampleTransform::new(0.0001, 0.0).unwrap(), None)
                    .unwrap(),
                RasterBandSemantics::new("nir", SampleTransform::new(0.0001, 0.0).unwrap(), None)
                    .unwrap(),
            ],
            ValidityMaskSemantics::None,
            24,
            ContentDigest::new(DigestAlgorithm::Sha256, hex256(digest_char)).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn joins_authoritative_window_and_payload() {
        let fixture = fixture();
        let window = window(&fixture);
        let payload = payload(RasterByteOrder::LittleEndian, '2');
        let receipt = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload).unwrap();
        receipt.verify_against(&fixture, &window, &payload).unwrap();
    }

    #[test]
    fn same_shape_but_different_bytes_fail() {
        let fixture = fixture();
        let window = window(&fixture);
        let payload = payload(RasterByteOrder::LittleEndian, '3');
        let err = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload).unwrap_err();
        assert_eq!(err, CanonicalRasterError::ContentDigestMismatch);
    }

    #[test]
    fn same_bytes_but_different_interpretation_have_different_semantics() {
        let fixture = fixture();
        let window = window(&fixture);
        let little = payload(RasterByteOrder::LittleEndian, '2');
        let big = payload(RasterByteOrder::BigEndian, '2');
        let first = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &little).unwrap();
        let second = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &big).unwrap();
        assert_ne!(first.payload_semantics_digest(), second.payload_semantics_digest());
        assert_ne!(first.receipt_digest(), second.receipt_digest());
    }

    #[test]
    fn shape_mismatch_fails_closed() {
        let fixture = fixture();
        let window = window(&fixture);
        let payload = RasterPayloadDescriptor::new(
            RasterShape::new(1, 6).unwrap(),
            RasterSampleType::U16,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandInterleavedByPixel,
            vec![
                RasterBandSemantics::new("red", SampleTransform::IDENTITY, None).unwrap(),
                RasterBandSemantics::new("nir", SampleTransform::IDENTITY, None).unwrap(),
            ],
            ValidityMaskSemantics::None,
            24,
            ContentDigest::new(DigestAlgorithm::Sha256, hex256('2')).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload),
            Err(CanonicalRasterError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn persisted_receipt_requires_authoritative_revalidation() {
        let fixture = fixture();
        let window = window(&fixture);
        let payload = payload(RasterByteOrder::LittleEndian, '2');
        let mut receipt = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload).unwrap();
        receipt.rows = 99;
        receipt.receipt_digest = receipt.compute_digest().unwrap();
        let value = serde_json::to_value(&receipt).unwrap();
        let loaded: CanonicalRasterEvidence = serde_json::from_value(value).unwrap();
        assert_eq!(
            loaded.verify_against(&fixture, &window, &payload),
            Err(CanonicalRasterError::AuthoritativeMismatch)
        );
    }
}
