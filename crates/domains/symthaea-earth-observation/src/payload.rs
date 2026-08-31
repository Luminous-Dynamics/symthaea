//! Canonical in-memory raster payload semantics for reproducible Earth observation.
//!
//! This module deliberately describes a decoded, packed scientific buffer. It
//! does **not** parse GeoTIFF/JP2/SAFE products, decompress provider files,
//! reproject, resample, or infer missing metadata. Provider I/O must materialize
//! one canonical buffer first, then bind that buffer to these semantics.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::raster::RasterShape;
use crate::ContentDigest;

pub type PayloadResult<T> = std::result::Result<T, RasterPayloadError>;

#[derive(Debug, Clone, PartialEq)]
pub enum RasterPayloadError {
    EmptyBandSet,
    TooManyBands(usize),
    EmptyBandName,
    DuplicateBandName(String),
    NonFiniteTransform { field: &'static str, value: f64 },
    InvalidByteOrder { sample_type: RasterSampleType, byte_order: RasterByteOrder },
    NoDataTypeMismatch { sample_type: RasterSampleType, nodata: NoDataValue },
    NoDataOutOfRange { sample_type: RasterSampleType, nodata: NoDataValue },
    EmbeddedMaskBandOutOfRange { band_index: u16, band_count: u16 },
    EmptyExternalMaskId,
    ArithmeticOverflow(&'static str),
    ByteLengthMismatch { expected: u64, actual: u64 },
    InvalidContentDigest,
    SampleOutOfBounds { row: u32, col: u32, band: u16 },
}

impl Display for RasterPayloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBandSet => write!(f, "raster payload must contain at least one band"),
            Self::TooManyBands(count) => write!(f, "raster payload band count {count} exceeds u16::MAX"),
            Self::EmptyBandName => write!(f, "raster band name must not be empty"),
            Self::DuplicateBandName(name) => write!(f, "duplicate raster band name: {name}"),
            Self::NonFiniteTransform { field, value } => {
                write!(f, "band transform field {field} must be finite, got {value}")
            }
            Self::InvalidByteOrder { sample_type, byte_order } => write!(
                f,
                "byte order {byte_order:?} is not canonical for sample type {sample_type:?}"
            ),
            Self::NoDataTypeMismatch { sample_type, nodata } => write!(
                f,
                "NoData value {nodata:?} does not match sample type {sample_type:?}"
            ),
            Self::NoDataOutOfRange { sample_type, nodata } => write!(
                f,
                "NoData value {nodata:?} is outside the range of sample type {sample_type:?}"
            ),
            Self::EmbeddedMaskBandOutOfRange { band_index, band_count } => write!(
                f,
                "embedded validity-mask band {band_index} is outside band count {band_count}"
            ),
            Self::EmptyExternalMaskId => write!(f, "external validity-mask id must not be empty"),
            Self::ArithmeticOverflow(field) => write!(f, "arithmetic overflow while computing {field}"),
            Self::ByteLengthMismatch { expected, actual } => write!(
                f,
                "canonical packed payload requires {expected} bytes, got {actual}"
            ),
            Self::InvalidContentDigest => write!(f, "payload content digest must be non-empty hexadecimal text"),
            Self::SampleOutOfBounds { row, col, band } => write!(
                f,
                "sample row={row} col={col} band={band} is outside the payload"
            ),
        }
    }
}

impl Error for RasterPayloadError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RasterSampleType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
}

impl RasterSampleType {
    pub const fn bytes_per_sample(self) -> u64 {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    const fn is_single_byte(self) -> bool {
        matches!(self, Self::U8 | Self::I8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RasterByteOrder {
    /// Canonical for one-byte sample types only.
    NotApplicable,
    LittleEndian,
    BigEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandInterleave {
    /// All pixels for band 0, then all pixels for band 1, etc. (BSQ).
    BandSequential,
    /// For each row, all columns of band 0, then all columns of band 1, etc. (BIL).
    BandInterleavedByLine,
    /// For each pixel, all bands are adjacent. (BIP).
    BandInterleavedByPixel,
}

/// Exact stored-value sentinel. Floating-point sentinels use raw IEEE bits so
/// a specific NaN payload, signed zero, or other bit pattern remains explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoDataValue {
    Unsigned(u64),
    Signed(i64),
    Float32Bits(u32),
    Float64Bits(u64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleTransform {
    /// Physical value = stored value * scale + offset.
    pub scale: f64,
    pub offset: f64,
}

impl SampleTransform {
    pub const IDENTITY: Self = Self { scale: 1.0, offset: 0.0 };

    pub fn new(scale: f64, offset: f64) -> PayloadResult<Self> {
        if !scale.is_finite() {
            return Err(RasterPayloadError::NonFiniteTransform { field: "scale", value: scale });
        }
        if !offset.is_finite() {
            return Err(RasterPayloadError::NonFiniteTransform { field: "offset", value: offset });
        }
        Ok(Self { scale, offset })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterBandSemantics {
    name: String,
    transform: SampleTransform,
    nodata: Option<NoDataValue>,
}

impl RasterBandSemantics {
    pub fn new(
        name: impl Into<String>,
        transform: SampleTransform,
        nodata: Option<NoDataValue>,
    ) -> PayloadResult<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RasterPayloadError::EmptyBandName);
        }
        if !transform.scale.is_finite() {
            return Err(RasterPayloadError::NonFiniteTransform {
                field: "scale",
                value: transform.scale,
            });
        }
        if !transform.offset.is_finite() {
            return Err(RasterPayloadError::NonFiniteTransform {
                field: "offset",
                value: transform.offset,
            });
        }
        Ok(Self { name, transform, nodata })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn transform(&self) -> SampleTransform {
        self.transform
    }

    pub const fn nodata(&self) -> Option<NoDataValue> {
        self.nodata
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityMaskSemantics {
    None,
    /// One band in the canonical payload acts as a validity mask.
    EmbeddedBand {
        band_index: u16,
        valid_when_nonzero: bool,
    },
    /// A separately content-addressed mask artifact is required by the caller.
    /// The evidence layer owns the actual artifact identity and bytes.
    External {
        mask_id: String,
        valid_when_nonzero: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterPayloadDescriptor {
    shape: RasterShape,
    sample_type: RasterSampleType,
    byte_order: RasterByteOrder,
    interleave: BandInterleave,
    bands: Vec<RasterBandSemantics>,
    validity_mask: ValidityMaskSemantics,
    byte_len: u64,
    content_digest: ContentDigest,
}

impl RasterPayloadDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shape: RasterShape,
        sample_type: RasterSampleType,
        byte_order: RasterByteOrder,
        interleave: BandInterleave,
        bands: Vec<RasterBandSemantics>,
        validity_mask: ValidityMaskSemantics,
        byte_len: u64,
        content_digest: ContentDigest,
    ) -> PayloadResult<Self> {
        if bands.is_empty() {
            return Err(RasterPayloadError::EmptyBandSet);
        }
        if bands.len() > u16::MAX as usize {
            return Err(RasterPayloadError::TooManyBands(bands.len()));
        }

        let mut names = HashSet::with_capacity(bands.len());
        for band in &bands {
            if band.name.trim().is_empty() {
                return Err(RasterPayloadError::EmptyBandName);
            }
            if !names.insert(band.name.clone()) {
                return Err(RasterPayloadError::DuplicateBandName(band.name.clone()));
            }
            if let Some(nodata) = band.nodata {
                validate_nodata(sample_type, nodata)?;
            }
        }

        if sample_type.is_single_byte() {
            if byte_order != RasterByteOrder::NotApplicable {
                return Err(RasterPayloadError::InvalidByteOrder { sample_type, byte_order });
            }
        } else if byte_order == RasterByteOrder::NotApplicable {
            return Err(RasterPayloadError::InvalidByteOrder { sample_type, byte_order });
        }

        let band_count = bands.len() as u16;
        match &validity_mask {
            ValidityMaskSemantics::EmbeddedBand { band_index, .. } if *band_index >= band_count => {
                return Err(RasterPayloadError::EmbeddedMaskBandOutOfRange {
                    band_index: *band_index,
                    band_count,
                });
            }
            ValidityMaskSemantics::External { mask_id, .. } if mask_id.trim().is_empty() => {
                return Err(RasterPayloadError::EmptyExternalMaskId);
            }
            _ => {}
        }

        if content_digest.hex.is_empty()
            || !content_digest.hex.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(RasterPayloadError::InvalidContentDigest);
        }

        let expected = expected_packed_bytes(shape, band_count, sample_type)?;
        if expected != byte_len {
            return Err(RasterPayloadError::ByteLengthMismatch {
                expected,
                actual: byte_len,
            });
        }

        Ok(Self {
            shape,
            sample_type,
            byte_order,
            interleave,
            bands,
            validity_mask,
            byte_len,
            content_digest,
        })
    }

    pub const fn shape(&self) -> RasterShape {
        self.shape
    }

    pub const fn sample_type(&self) -> RasterSampleType {
        self.sample_type
    }

    pub const fn byte_order(&self) -> RasterByteOrder {
        self.byte_order
    }

    pub const fn interleave(&self) -> BandInterleave {
        self.interleave
    }

    pub fn bands(&self) -> &[RasterBandSemantics] {
        &self.bands
    }

    pub const fn band_count(&self) -> u16 {
        self.bands.len() as u16
    }

    pub fn validity_mask(&self) -> &ValidityMaskSemantics {
        &self.validity_mask
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn content_digest(&self) -> &ContentDigest {
        &self.content_digest
    }

    /// Byte offset of one stored sample within the canonical packed buffer.
    pub fn sample_offset(&self, row: u32, col: u32, band: u16) -> PayloadResult<u64> {
        if row >= self.shape.rows() || col >= self.shape.cols() || band >= self.band_count() {
            return Err(RasterPayloadError::SampleOutOfBounds { row, col, band });
        }

        let rows = self.shape.rows() as u64;
        let cols = self.shape.cols() as u64;
        let bands = self.band_count() as u64;
        let row = row as u64;
        let col = col as u64;
        let band = band as u64;

        let sample_index = match self.interleave {
            BandInterleave::BandSequential => band
                .checked_mul(rows)
                .and_then(|value| value.checked_add(row))
                .and_then(|value| value.checked_mul(cols))
                .and_then(|value| value.checked_add(col))
                .ok_or(RasterPayloadError::ArithmeticOverflow("BSQ sample index"))?,
            BandInterleave::BandInterleavedByLine => row
                .checked_mul(bands)
                .and_then(|value| value.checked_add(band))
                .and_then(|value| value.checked_mul(cols))
                .and_then(|value| value.checked_add(col))
                .ok_or(RasterPayloadError::ArithmeticOverflow("BIL sample index"))?,
            BandInterleave::BandInterleavedByPixel => row
                .checked_mul(cols)
                .and_then(|value| value.checked_add(col))
                .and_then(|value| value.checked_mul(bands))
                .and_then(|value| value.checked_add(band))
                .ok_or(RasterPayloadError::ArithmeticOverflow("BIP sample index"))?,
        };

        sample_index
            .checked_mul(self.sample_type.bytes_per_sample())
            .ok_or(RasterPayloadError::ArithmeticOverflow("sample byte offset"))
    }
}

fn expected_packed_bytes(
    shape: RasterShape,
    band_count: u16,
    sample_type: RasterSampleType,
) -> PayloadResult<u64> {
    shape
        .pixel_count()
        .checked_mul(band_count as u64)
        .and_then(|value| value.checked_mul(sample_type.bytes_per_sample()))
        .ok_or(RasterPayloadError::ArithmeticOverflow("packed payload byte length"))
}

fn validate_nodata(sample_type: RasterSampleType, nodata: NoDataValue) -> PayloadResult<()> {
    let type_matches = matches!(
        (sample_type, nodata),
        (RasterSampleType::U8 | RasterSampleType::U16 | RasterSampleType::U32, NoDataValue::Unsigned(_))
            | (RasterSampleType::I8 | RasterSampleType::I16 | RasterSampleType::I32, NoDataValue::Signed(_))
            | (RasterSampleType::F32, NoDataValue::Float32Bits(_))
            | (RasterSampleType::F64, NoDataValue::Float64Bits(_))
    );
    if !type_matches {
        return Err(RasterPayloadError::NoDataTypeMismatch { sample_type, nodata });
    }

    let in_range = match (sample_type, nodata) {
        (RasterSampleType::U8, NoDataValue::Unsigned(value)) => value <= u8::MAX as u64,
        (RasterSampleType::U16, NoDataValue::Unsigned(value)) => value <= u16::MAX as u64,
        (RasterSampleType::U32, NoDataValue::Unsigned(value)) => value <= u32::MAX as u64,
        (RasterSampleType::I8, NoDataValue::Signed(value)) => i8::try_from(value).is_ok(),
        (RasterSampleType::I16, NoDataValue::Signed(value)) => i16::try_from(value).is_ok(),
        (RasterSampleType::I32, NoDataValue::Signed(value)) => i32::try_from(value).is_ok(),
        (RasterSampleType::F32, NoDataValue::Float32Bits(_))
        | (RasterSampleType::F64, NoDataValue::Float64Bits(_)) => true,
        _ => false,
    };
    if !in_range {
        return Err(RasterPayloadError::NoDataOutOfRange { sample_type, nodata });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DigestAlgorithm, ContentDigest};

    fn digest() -> ContentDigest {
        ContentDigest::new(DigestAlgorithm::Sha256, "11".repeat(32)).unwrap()
    }

    fn bands() -> Vec<RasterBandSemantics> {
        vec![
            RasterBandSemantics::new(
                "red",
                SampleTransform::new(0.0001, 0.0).unwrap(),
                Some(NoDataValue::Unsigned(0)),
            )
            .unwrap(),
            RasterBandSemantics::new(
                "nir",
                SampleTransform::new(0.0001, 0.0).unwrap(),
                Some(NoDataValue::Unsigned(0)),
            )
            .unwrap(),
        ]
    }

    fn descriptor(interleave: BandInterleave) -> RasterPayloadDescriptor {
        RasterPayloadDescriptor::new(
            RasterShape::new(2, 3).unwrap(),
            RasterSampleType::U16,
            RasterByteOrder::LittleEndian,
            interleave,
            bands(),
            ValidityMaskSemantics::None,
            24,
            digest(),
        )
        .unwrap()
    }

    #[test]
    fn canonical_packed_layouts_have_distinct_offsets() {
        let bsq = descriptor(BandInterleave::BandSequential);
        let bil = descriptor(BandInterleave::BandInterleavedByLine);
        let bip = descriptor(BandInterleave::BandInterleavedByPixel);

        assert_eq!(bsq.sample_offset(0, 1, 1).unwrap(), 14);
        assert_eq!(bil.sample_offset(0, 1, 1).unwrap(), 8);
        assert_eq!(bip.sample_offset(0, 1, 1).unwrap(), 6);
    }

    #[test]
    fn byte_order_is_canonical_for_sample_width() {
        let one_byte = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::U8,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandSequential,
            vec![RasterBandSemantics::new("mask", SampleTransform::IDENTITY, None).unwrap()],
            ValidityMaskSemantics::None,
            1,
            digest(),
        )
        .unwrap_err();
        assert!(matches!(one_byte, RasterPayloadError::InvalidByteOrder { .. }));

        let multi_byte = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::F32,
            RasterByteOrder::NotApplicable,
            BandInterleave::BandSequential,
            vec![RasterBandSemantics::new("value", SampleTransform::IDENTITY, None).unwrap()],
            ValidityMaskSemantics::None,
            4,
            digest(),
        )
        .unwrap_err();
        assert!(matches!(multi_byte, RasterPayloadError::InvalidByteOrder { .. }));
    }

    #[test]
    fn byte_length_must_match_exact_packed_shape() {
        let err = RasterPayloadDescriptor::new(
            RasterShape::new(2, 3).unwrap(),
            RasterSampleType::U16,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandSequential,
            bands(),
            ValidityMaskSemantics::None,
            23,
            digest(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            RasterPayloadError::ByteLengthMismatch {
                expected: 24,
                actual: 23,
            }
        );
    }

    #[test]
    fn nodata_must_match_storage_type_and_range() {
        let mismatch = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::U16,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandSequential,
            vec![RasterBandSemantics::new(
                "value",
                SampleTransform::IDENTITY,
                Some(NoDataValue::Signed(-1)),
            )
            .unwrap()],
            ValidityMaskSemantics::None,
            2,
            digest(),
        )
        .unwrap_err();
        assert!(matches!(mismatch, RasterPayloadError::NoDataTypeMismatch { .. }));

        let range = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::U8,
            RasterByteOrder::NotApplicable,
            BandInterleave::BandSequential,
            vec![RasterBandSemantics::new(
                "value",
                SampleTransform::IDENTITY,
                Some(NoDataValue::Unsigned(256)),
            )
            .unwrap()],
            ValidityMaskSemantics::None,
            1,
            digest(),
        )
        .unwrap_err();
        assert!(matches!(range, RasterPayloadError::NoDataOutOfRange { .. }));
    }

    #[test]
    fn validity_mask_semantics_are_explicit_and_checked() {
        let embedded = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::U16,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandSequential,
            bands(),
            ValidityMaskSemantics::EmbeddedBand {
                band_index: 2,
                valid_when_nonzero: true,
            },
            4,
            digest(),
        )
        .unwrap_err();
        assert!(matches!(
            embedded,
            RasterPayloadError::EmbeddedMaskBandOutOfRange { .. }
        ));

        let external = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::U16,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandSequential,
            bands(),
            ValidityMaskSemantics::External {
                mask_id: " ".to_string(),
                valid_when_nonzero: true,
            },
            4,
            digest(),
        )
        .unwrap_err();
        assert_eq!(external, RasterPayloadError::EmptyExternalMaskId);
    }

    #[test]
    fn duplicate_band_names_fail_closed() {
        let duplicate = vec![
            RasterBandSemantics::new("red", SampleTransform::IDENTITY, None).unwrap(),
            RasterBandSemantics::new("red", SampleTransform::IDENTITY, None).unwrap(),
        ];
        let err = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::U8,
            RasterByteOrder::NotApplicable,
            BandInterleave::BandSequential,
            duplicate,
            ValidityMaskSemantics::None,
            2,
            digest(),
        )
        .unwrap_err();
        assert_eq!(err, RasterPayloadError::DuplicateBandName("red".to_string()));
    }

    #[test]
    fn out_of_bounds_sample_lookup_fails_closed() {
        let payload = descriptor(BandInterleave::BandInterleavedByPixel);
        let err = payload.sample_offset(2, 0, 0).unwrap_err();
        assert!(matches!(err, RasterPayloadError::SampleOutOfBounds { .. }));
    }

    #[test]
    fn floating_nodata_preserves_exact_bits() {
        let nodata_bits = f32::NAN.to_bits();
        let payload = RasterPayloadDescriptor::new(
            RasterShape::new(1, 1).unwrap(),
            RasterSampleType::F32,
            RasterByteOrder::LittleEndian,
            BandInterleave::BandSequential,
            vec![RasterBandSemantics::new(
                "temperature",
                SampleTransform::IDENTITY,
                Some(NoDataValue::Float32Bits(nodata_bits)),
            )
            .unwrap()],
            ValidityMaskSemantics::None,
            4,
            digest(),
        )
        .unwrap();
        assert_eq!(payload.bands()[0].nodata(), Some(NoDataValue::Float32Bits(nodata_bits)));
    }
}
