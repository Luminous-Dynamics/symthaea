//! Deterministic SAR backscatter semantics for Planetary Perception.
//!
//! This module intentionally stops before interferometry, phase unwrapping,
//! terrain correction, speckle filtering, radiometric calibration, or any claim
//! about subsurface penetration. Provider/processor code must perform and record
//! those steps before constructing these calibrated samples.

use crate::{Confidence, EvidenceError, Result};
use crate::features::MaskReason;

/// Backscatter values are not interchangeable across these scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackscatterScale {
    /// Linear power-like backscatter coefficient (strictly positive for dB conversion).
    LinearPower,
    /// `10 * log10(linear_power)`.
    Decibel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackscatterSample {
    Valid {
        value: f64,
        scale: BackscatterScale,
        quality: Option<Confidence>,
    },
    Masked(MaskReason),
}

impl BackscatterSample {
    pub fn valid(
        value: f64,
        scale: BackscatterScale,
        quality: Option<Confidence>,
    ) -> Result<Self> {
        if !value.is_finite() {
            return Err(EvidenceError::NonFinite("SAR backscatter sample"));
        }
        if scale == BackscatterScale::LinearPower && value < 0.0 {
            return Err(EvidenceError::Negative("linear SAR backscatter", value));
        }
        Ok(Self::Valid {
            value,
            scale,
            quality,
        })
    }

    pub const fn masked(reason: MaskReason) -> Self {
        Self::Masked(reason)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SarFeatureKind {
    /// Linear power ratio, e.g. VV / VH.
    PolarizationRatioLinear,
    /// Difference in decibels, e.g. VV_dB - VH_dB.
    PolarizationDifferenceDb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SarFeatureStatus {
    Valid,
    MaskedInput(MaskReason),
    NonPositiveLinearInput,
    ZeroDenominator,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SarFeatureSample {
    pub kind: SarFeatureKind,
    pub value: Option<f64>,
    pub quality: Option<Confidence>,
    pub status: SarFeatureStatus,
}

impl SarFeatureSample {
    pub const fn is_valid(self) -> bool {
        matches!(self.status, SarFeatureStatus::Valid)
    }
}

fn combined_quality(a: Option<Confidence>, b: Option<Confidence>) -> Option<Confidence> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a.get() <= b.get() { a } else { b }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Convert a strictly positive linear-power backscatter value to decibels.
///
/// Zero is not converted to `-inf`; it is represented as an invalid/masked
/// measurement by the caller. That keeps non-finite values out of the evidence
/// graph.
pub fn linear_power_to_db(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(EvidenceError::NonFinite("linear SAR backscatter"));
    }
    if value < 0.0 {
        return Err(EvidenceError::Negative("linear SAR backscatter", value));
    }
    if value == 0.0 {
        return Err(EvidenceError::NonFinite(
            "logarithm of zero linear SAR backscatter",
        ));
    }
    Ok(10.0 * value.log10())
}

/// Convert decibels to linear power.
pub fn db_to_linear_power(value_db: f64) -> Result<f64> {
    if !value_db.is_finite() {
        return Err(EvidenceError::NonFinite("decibel SAR backscatter"));
    }
    let value = 10.0_f64.powf(value_db / 10.0);
    if !value.is_finite() {
        return Err(EvidenceError::NonFinite(
            "converted linear SAR backscatter",
        ));
    }
    Ok(value)
}

pub fn to_db(sample: BackscatterSample) -> Result<BackscatterSample> {
    match sample {
        BackscatterSample::Masked(reason) => Ok(BackscatterSample::Masked(reason)),
        BackscatterSample::Valid {
            value,
            scale: BackscatterScale::Decibel,
            quality,
        } => Ok(BackscatterSample::Valid {
            value,
            scale: BackscatterScale::Decibel,
            quality,
        }),
        BackscatterSample::Valid {
            value,
            scale: BackscatterScale::LinearPower,
            quality,
        } => Ok(BackscatterSample::Valid {
            value: linear_power_to_db(value)?,
            scale: BackscatterScale::Decibel,
            quality,
        }),
    }
}

pub fn to_linear_power(sample: BackscatterSample) -> Result<BackscatterSample> {
    match sample {
        BackscatterSample::Masked(reason) => Ok(BackscatterSample::Masked(reason)),
        BackscatterSample::Valid {
            value,
            scale: BackscatterScale::LinearPower,
            quality,
        } => Ok(BackscatterSample::Valid {
            value,
            scale: BackscatterScale::LinearPower,
            quality,
        }),
        BackscatterSample::Valid {
            value,
            scale: BackscatterScale::Decibel,
            quality,
        } => Ok(BackscatterSample::Valid {
            value: db_to_linear_power(value)?,
            scale: BackscatterScale::LinearPower,
            quality,
        }),
    }
}

fn unpack_pair(
    numerator: BackscatterSample,
    denominator: BackscatterSample,
) -> Result<std::result::Result<(f64, f64, Option<Confidence>), MaskReason>> {
    let numerator = to_linear_power(numerator)?;
    let denominator = to_linear_power(denominator)?;

    let (a, qa) = match numerator {
        BackscatterSample::Masked(reason) => return Ok(Err(reason)),
        BackscatterSample::Valid { value, quality, .. } => (value, quality),
    };
    let (b, qb) = match denominator {
        BackscatterSample::Masked(reason) => return Ok(Err(reason)),
        BackscatterSample::Valid { value, quality, .. } => (value, quality),
    };
    Ok(Ok((a, b, combined_quality(qa, qb))))
}

/// Compute a polarization power ratio such as `VV / VH` in linear space.
///
/// Even if the input samples are supplied in dB, the ratio is evaluated in
/// linear space and labelled accordingly.
pub fn polarization_ratio_linear(
    numerator: BackscatterSample,
    denominator: BackscatterSample,
) -> Result<SarFeatureSample> {
    let (numerator, denominator, quality) = match unpack_pair(numerator, denominator)? {
        Ok(values) => values,
        Err(reason) => {
            return Ok(SarFeatureSample {
                kind: SarFeatureKind::PolarizationRatioLinear,
                value: None,
                quality: None,
                status: SarFeatureStatus::MaskedInput(reason),
            });
        }
    };

    if numerator <= 0.0 || denominator < 0.0 {
        return Ok(SarFeatureSample {
            kind: SarFeatureKind::PolarizationRatioLinear,
            value: None,
            quality,
            status: SarFeatureStatus::NonPositiveLinearInput,
        });
    }
    if denominator == 0.0 {
        return Ok(SarFeatureSample {
            kind: SarFeatureKind::PolarizationRatioLinear,
            value: None,
            quality,
            status: SarFeatureStatus::ZeroDenominator,
        });
    }

    Ok(SarFeatureSample {
        kind: SarFeatureKind::PolarizationRatioLinear,
        value: Some(numerator / denominator),
        quality,
        status: SarFeatureStatus::Valid,
    })
}

/// Compute the polarization difference in dB, e.g. `VV_dB - VH_dB`.
///
/// This quantity is related to, but not numerically interchangeable with, the
/// linear polarization ratio. Keeping separate API/type identities prevents a
/// downstream model from accidentally mixing the two feature spaces.
pub fn polarization_difference_db(
    minuend: BackscatterSample,
    subtrahend: BackscatterSample,
) -> Result<SarFeatureSample> {
    let minuend = to_db(minuend)?;
    let subtrahend = to_db(subtrahend)?;

    let (a, qa) = match minuend {
        BackscatterSample::Masked(reason) => {
            return Ok(SarFeatureSample {
                kind: SarFeatureKind::PolarizationDifferenceDb,
                value: None,
                quality: None,
                status: SarFeatureStatus::MaskedInput(reason),
            });
        }
        BackscatterSample::Valid { value, quality, .. } => (value, quality),
    };
    let (b, qb) = match subtrahend {
        BackscatterSample::Masked(reason) => {
            return Ok(SarFeatureSample {
                kind: SarFeatureKind::PolarizationDifferenceDb,
                value: None,
                quality: None,
                status: SarFeatureStatus::MaskedInput(reason),
            });
        }
        BackscatterSample::Valid { value, quality, .. } => (value, quality),
    };

    Ok(SarFeatureSample {
        kind: SarFeatureKind::PolarizationDifferenceDb,
        value: Some(a - b),
        quality: combined_quality(qa, qb),
        status: SarFeatureStatus::Valid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(value: f64) -> BackscatterSample {
        BackscatterSample::valid(value, BackscatterScale::LinearPower, None).unwrap()
    }

    fn db(value: f64) -> BackscatterSample {
        BackscatterSample::valid(value, BackscatterScale::Decibel, None).unwrap()
    }

    #[test]
    fn db_linear_roundtrip_is_stable() {
        let original = 0.03125;
        let db = linear_power_to_db(original).unwrap();
        let reconstructed = db_to_linear_power(db).unwrap();
        assert!((original - reconstructed).abs() < 1.0e-12);
    }

    #[test]
    fn zero_linear_backscatter_never_becomes_negative_infinity() {
        assert!(linear_power_to_db(0.0).is_err());
    }

    #[test]
    fn polarization_ratio_uses_linear_power() {
        let result = polarization_ratio_linear(linear(0.4), linear(0.1)).unwrap();
        assert!(result.is_valid());
        assert_eq!(result.kind, SarFeatureKind::PolarizationRatioLinear);
        assert!((result.value.unwrap() - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn db_difference_matches_ratio_identity() {
        let ratio = polarization_ratio_linear(linear(0.4), linear(0.1))
            .unwrap()
            .value
            .unwrap();
        let difference = polarization_difference_db(linear(0.4), linear(0.1))
            .unwrap()
            .value
            .unwrap();
        assert!((difference - 10.0 * ratio.log10()).abs() < 1.0e-12);
    }

    #[test]
    fn masked_sample_propagates() {
        let result = polarization_ratio_linear(
            BackscatterSample::masked(MaskReason::NoData),
            linear(0.1),
        )
        .unwrap();
        assert_eq!(result.value, None);
        assert_eq!(result.status, SarFeatureStatus::MaskedInput(MaskReason::NoData));
    }

    #[test]
    fn db_inputs_are_not_treated_as_linear_values() {
        let result = polarization_ratio_linear(db(-10.0), db(-20.0)).unwrap();
        assert!((result.value.unwrap() - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn quality_propagates_conservatively() {
        let high = BackscatterSample::valid(
            0.4,
            BackscatterScale::LinearPower,
            Some(Confidence::new(0.93).unwrap()),
        )
        .unwrap();
        let low = BackscatterSample::valid(
            0.1,
            BackscatterScale::LinearPower,
            Some(Confidence::new(0.68).unwrap()),
        )
        .unwrap();
        let result = polarization_ratio_linear(high, low).unwrap();
        assert_eq!(result.quality.unwrap().get(), 0.68);
    }
}
