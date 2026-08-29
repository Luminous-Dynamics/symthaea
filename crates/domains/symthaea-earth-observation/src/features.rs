//! Deterministic, provider-neutral Earth-observation feature math.
//!
//! These functions operate on calibrated scalar samples. Raster traversal,
//! resampling, cloud classification, atmospheric correction, and provider I/O
//! belong elsewhere. The goal here is to make the arithmetic and failure
//! semantics explicit enough to replay exactly in tests and evidence lineages.

use crate::{Confidence, EvidenceError, Result, SpectralIndex};

/// Default denominator floor for normalized-difference indices.
///
/// This is a numerical guard, not a sensor-noise model. Domain/provider code may
/// choose a larger, physically justified floor via `normalized_difference`.
pub const DEFAULT_DENOMINATOR_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskReason {
    Cloud,
    CloudShadow,
    SnowOrIce,
    Saturated,
    NoData,
    OutsideFootprint,
    ProviderQualityFlag,
    Other,
}

/// One already-calibrated band sample.
///
/// A masked sample carries no numeric value by design. This prevents callers
/// from accidentally computing an index from a placeholder such as zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandSample {
    Valid {
        value: f64,
        quality: Option<Confidence>,
    },
    Masked(MaskReason),
}

impl BandSample {
    pub fn valid(value: f64, quality: Option<Confidence>) -> Result<Self> {
        if !value.is_finite() {
            return Err(EvidenceError::NonFinite("band sample"));
        }
        Ok(Self::Valid { value, quality })
    }

    pub const fn masked(reason: MaskReason) -> Self {
        Self::Masked(reason)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexStatus {
    Valid,
    MaskedInput(MaskReason),
    DegenerateDenominator,
}

/// One normalized-difference result with explicit validity semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexSample {
    pub index: SpectralIndex,
    pub value: Option<f64>,
    /// Conservative input-quality propagation: minimum of all supplied
    /// confidences. If neither input carries a quality score, this stays None.
    pub quality: Option<Confidence>,
    pub status: IndexStatus,
}

impl IndexSample {
    pub const fn is_valid(self) -> bool {
        matches!(self.status, IndexStatus::Valid)
    }
}

fn combined_quality(a: Option<Confidence>, b: Option<Confidence>) -> Option<Confidence> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a.get() <= b.get() { a } else { b }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn masked_result(index: SpectralIndex, reason: MaskReason) -> IndexSample {
    IndexSample {
        index,
        value: None,
        quality: None,
        status: IndexStatus::MaskedInput(reason),
    }
}

/// Compute `(positive - negative) / (positive + negative)` with explicit mask
/// and near-zero-denominator handling.
///
/// `epsilon` must be finite and non-negative. No output clamp is applied: if a
/// caller supplies values outside the physically expected range, preserving the
/// result is more auditable than silently rewriting it.
pub fn normalized_difference(
    index: SpectralIndex,
    positive: BandSample,
    negative: BandSample,
    epsilon: f64,
) -> Result<IndexSample> {
    if !epsilon.is_finite() {
        return Err(EvidenceError::NonFinite("normalized-difference epsilon"));
    }
    if epsilon < 0.0 {
        return Err(EvidenceError::Negative(
            "normalized-difference epsilon",
            epsilon,
        ));
    }

    let (positive_value, positive_quality) = match positive {
        BandSample::Valid { value, quality } => (value, quality),
        BandSample::Masked(reason) => return Ok(masked_result(index, reason)),
    };
    let (negative_value, negative_quality) = match negative {
        BandSample::Valid { value, quality } => (value, quality),
        BandSample::Masked(reason) => return Ok(masked_result(index, reason)),
    };

    let denominator = positive_value + negative_value;
    let quality = combined_quality(positive_quality, negative_quality);
    if denominator.abs() <= epsilon {
        return Ok(IndexSample {
            index,
            value: None,
            quality,
            status: IndexStatus::DegenerateDenominator,
        });
    }

    Ok(IndexSample {
        index,
        value: Some((positive_value - negative_value) / denominator),
        quality,
        status: IndexStatus::Valid,
    })
}

/// Normalized Difference Vegetation Index: `(NIR - Red) / (NIR + Red)`.
pub fn ndvi(nir: BandSample, red: BandSample) -> Result<IndexSample> {
    normalized_difference(SpectralIndex::Ndvi, nir, red, DEFAULT_DENOMINATOR_EPSILON)
}

/// McFeeters water index: `(Green - NIR) / (Green + NIR)`.
///
/// This is intentionally named separately from Gao's NDWI.
pub fn mcfeeters_ndwi(green: BandSample, nir: BandSample) -> Result<IndexSample> {
    normalized_difference(
        SpectralIndex::McFeetersNdwi,
        green,
        nir,
        DEFAULT_DENOMINATOR_EPSILON,
    )
}

/// Gao vegetation-water index: `(NIR - SWIR) / (NIR + SWIR)`.
///
/// This is intentionally named separately from McFeeters' NDWI.
pub fn gao_ndwi(nir: BandSample, swir: BandSample) -> Result<IndexSample> {
    normalized_difference(
        SpectralIndex::GaoNdwi,
        nir,
        swir,
        DEFAULT_DENOMINATOR_EPSILON,
    )
}

/// Normalized Burn Ratio: `(NIR - SWIR2) / (NIR + SWIR2)`.
pub fn nbr(nir: BandSample, swir2: BandSample) -> Result<IndexSample> {
    normalized_difference(SpectralIndex::Nbr, nir, swir2, DEFAULT_DENOMINATOR_EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(value: f64) -> BandSample {
        BandSample::valid(value, None).unwrap()
    }

    #[test]
    fn ndvi_matches_definition() {
        let result = ndvi(sample(0.8), sample(0.2)).unwrap();
        assert!(result.is_valid());
        assert_eq!(result.index, SpectralIndex::Ndvi);
        assert!((result.value.unwrap() - 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn ndwi_formulations_are_not_interchangeable() {
        let green = sample(0.4);
        let nir = sample(0.7);
        let swir = sample(0.1);

        let mcfeeters = mcfeeters_ndwi(green, nir).unwrap();
        let gao = gao_ndwi(nir, swir).unwrap();

        assert_eq!(mcfeeters.index, SpectralIndex::McFeetersNdwi);
        assert_eq!(gao.index, SpectralIndex::GaoNdwi);
        assert_ne!(mcfeeters.value, gao.value);
    }

    #[test]
    fn mask_propagates_without_placeholder_arithmetic() {
        let result = ndvi(BandSample::masked(MaskReason::Cloud), sample(0.2)).unwrap();
        assert_eq!(result.value, None);
        assert_eq!(result.status, IndexStatus::MaskedInput(MaskReason::Cloud));
    }

    #[test]
    fn near_zero_denominator_is_explicit() {
        let result = normalized_difference(
            SpectralIndex::Ndvi,
            sample(1.0),
            sample(-1.0),
            DEFAULT_DENOMINATOR_EPSILON,
        )
        .unwrap();
        assert_eq!(result.value, None);
        assert_eq!(result.status, IndexStatus::DegenerateDenominator);
    }

    #[test]
    fn quality_propagates_conservatively() {
        let high = BandSample::valid(0.8, Some(Confidence::new(0.95).unwrap())).unwrap();
        let lower = BandSample::valid(0.2, Some(Confidence::new(0.72).unwrap())).unwrap();
        let result = ndvi(high, lower).unwrap();
        assert_eq!(result.quality.unwrap().get(), 0.72);
    }

    #[test]
    fn nonfinite_sample_is_rejected_at_construction() {
        assert_eq!(
            BandSample::valid(f64::NAN, None),
            Err(EvidenceError::NonFinite("band sample"))
        );
        assert_eq!(
            BandSample::valid(f64::INFINITY, None),
            Err(EvidenceError::NonFinite("band sample"))
        );
    }

    #[test]
    fn invalid_epsilon_is_rejected() {
        assert!(normalized_difference(
            SpectralIndex::Ndvi,
            sample(0.8),
            sample(0.2),
            -1.0,
        )
        .is_err());
    }
}
