//! Evidence join between exact raster provenance and canonical payload semantics.
//!
//! This crate has one job: prove that an exact-window receipt and a canonical
//! raster-payload descriptor describe the same bytes and shape, while also
//! content-addressing the declared interpretation of those bytes.

use serde::{Deserialize, Serialize};
use symthaea_earth_observation::{
    payload::{
        BandInterleave, NoDataValue, RasterByteOrder, RasterPayloadDescriptor,
        RasterSampleType, ValidityMaskSemantics,
    },
    DigestAlgorithm,
};
use symthaea_sentinel_eo::{ExactPixelWindowEvidence, FrozenSentinelFixtureManifest};
use thiserror::Error;

const PAYLOAD_SEMANTICS_SCHEMA: &str = "symthaea-canonical-raster-payload-semantics/v1";
const RECEIPT_SCHEMA: &str = "symthaea-canonical-raster-evidence/v1";
const HEX_256_LEN: usize = 64;

pub type Result<T> = std::result::Result<T, RasterEvidenceError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RasterEvidenceError {
    #[error("exact-window verification failed: {0}")]
    Window(String),
    #[error("payload content digest is noncanonical: {0}")]
    NonCanonicalPayloadDigest(String),
    #[error("payload content digest does not match exact-window output bytes")]
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
    #[error("stored raster-evidence digest does not match recomputation")]
    ReceiptDigestMismatch,
    #[error("stored raster evidence does not match authoritative inputs")]
    AuthoritativeMismatch,
    #[error("raster-evidence serialization failed: {0}")]
    Serialization(String),
}

fn blake3_json<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| RasterEvidenceError::Serialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn canonical_hex(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !value.bytes().any(|byte| byte.is_ascii_uppercase())
}

fn digest_tag(value: &DigestAlgorithm) -> &'static str {
    match value {
        DigestAlgorithm::Sha256 => "sha256",
        DigestAlgorithm::Blake3 => "blake3",
        DigestAlgorithm::Other => "other",
    }
}

fn canonical_payload_digest(payload: &RasterPayloadDescriptor) -> Result<(&'static str, String)> {
    let digest = payload.content_digest();
    let algorithm = digest_tag(&digest.algorithm);
    if !canonical_hex(&digest.hex) {
        return Err(RasterEvidenceError::NonCanonicalPayloadDigest(
            "hex must be lowercase hexadecimal".into(),
        ));
    }
    if matches!(algorithm, "sha256" | "blake3") && digest.hex.len() != HEX_256_LEN {
        return Err(RasterEvidenceError::NonCanonicalPayloadDigest(format!(
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
struct PayloadIdentity {
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

/// Stable identity for payload interpretation, deliberately separate from the
/// payload bytes themselves.
pub fn payload_semantics_digest(payload: &RasterPayloadDescriptor) -> Result<String> {
    let shape = payload.shape();
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
    blake3_json(&PayloadIdentity {
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
struct Repr {
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
struct ReceiptView<'a> {
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
    ) -> Result<Self> {
        window
            .verify_against_fixture(fixture)
            .map_err(|error| RasterEvidenceError::Window(error.to_string()))?;
        let (algorithm, hex) = canonical_payload_digest(payload)?;
        if window.output_content_digest.algorithm != algorithm || window.output_content_digest.hex != hex {
            return Err(RasterEvidenceError::ContentDigestMismatch);
        }

        let shape = payload.shape();
        if shape.rows() != window.geometry.output_rows || shape.cols() != window.geometry.output_cols {
            return Err(RasterEvidenceError::ShapeMismatch {
                payload_rows: shape.rows(),
                payload_cols: shape.cols(),
                window_rows: window.geometry.output_rows,
                window_cols: window.geometry.output_cols,
            });
        }
        if let Some(window_len) = window.byte_len {
            if window_len != payload.byte_len() {
                return Err(RasterEvidenceError::ByteLengthMismatch {
                    payload: payload.byte_len(),
                    window: window_len,
                });
            }
        }

        let mut result = Self {
            exact_window_evidence_digest: window.evidence_digest.clone(),
            fixture_manifest_digest: window.fixture_manifest_digest.clone(),
            source_identity_digest: window.source_identity_digest.clone(),
            payload_semantics_digest: payload_semantics_digest(payload)?,
            content_algorithm: algorithm.to_string(),
            content_hex: hex,
            rows: shape.rows(),
            cols: shape.cols(),
            byte_len: payload.byte_len(),
            receipt_digest: String::new(),
        };
        result.receipt_digest = result.compute_digest()?;
        Ok(result)
    }

    fn digest_view(&self) -> ReceiptView<'_> {
        ReceiptView {
            schema: RECEIPT_SCHEMA,
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

    pub fn compute_digest(&self) -> Result<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> Result<()> {
        for value in [
            &self.exact_window_evidence_digest,
            &self.fixture_manifest_digest,
            &self.source_identity_digest,
            &self.payload_semantics_digest,
            &self.receipt_digest,
        ] {
            if !canonical_hex(value) || value.len() != HEX_256_LEN {
                return Err(RasterEvidenceError::NonCanonicalPayloadDigest(
                    "evidence identities must be 64 lowercase hexadecimal characters".into(),
                ));
            }
        }
        if !matches!(self.content_algorithm.as_str(), "sha256" | "blake3" | "other")
            || !canonical_hex(&self.content_hex)
            || (matches!(self.content_algorithm.as_str(), "sha256" | "blake3")
                && self.content_hex.len() != HEX_256_LEN)
        {
            return Err(RasterEvidenceError::NonCanonicalPayloadDigest(
                "receipt content digest is not canonical".into(),
            ));
        }
        if self.compute_digest()? != self.receipt_digest {
            return Err(RasterEvidenceError::ReceiptDigestMismatch);
        }
        Ok(())
    }

    pub fn verify_against(
        &self,
        fixture: &FrozenSentinelFixtureManifest,
        window: &ExactPixelWindowEvidence,
        payload: &RasterPayloadDescriptor,
    ) -> Result<()> {
        self.verify_digest()?;
        let expected = Self::new_for_fixture(fixture, window, payload)?;
        if self != &expected {
            return Err(RasterEvidenceError::AuthoritativeMismatch);
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

impl TryFrom<Repr> for CanonicalRasterEvidence {
    type Error = RasterEvidenceError;

    fn try_from(value: Repr) -> Result<Self> {
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
        let repr = Repr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}
