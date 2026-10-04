//! Explicit continuous/binary HDC codec boundary.
//!
//! This module is opt-in because the communication crate should remain
//! lightweight unless a deployment explicitly wants Symthaea HDC integration.
//! The codec measures quantization separately from semantic decoding.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const HDC_CODEC_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcQuantizationMetrics {
    pub schema_version: u16,
    pub dimension: usize,
    pub continuous_bytes: usize,
    pub binary_bytes: usize,
    pub compression_ratio: f64,
    pub cosine_similarity: f64,
    pub sign_disagreement_rate: f64,
}

impl HdcQuantizationMetrics {
    pub fn validates(&self) -> bool {
        self.schema_version == HDC_CODEC_SCHEMA_VERSION
            && self.dimension == HDC_DIMENSION
            && self.continuous_bytes > 0
            && self.binary_bytes > 0
            && self.compression_ratio.is_finite()
            && self.compression_ratio >= 1.0
            && self.cosine_similarity.is_finite()
            && (-1.0..=1.0).contains(&self.cosine_similarity)
            && self.sign_disagreement_rate.is_finite()
            && (0.0..=1.0).contains(&self.sign_disagreement_rate)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcBinaryFrame {
    pub schema_version: u16,
    pub dimension: usize,
    pub bits: Vec<u8>,
}

impl HdcBinaryFrame {
    pub fn from_binary(binary: &BinaryHV) -> Self {
        Self {
            schema_version: HDC_CODEC_SCHEMA_VERSION,
            dimension: BinaryHV::DIM,
            bits: binary.0.to_vec(),
        }
    }

    pub fn to_binary(&self) -> Result<BinaryHV, String> {
        if self.schema_version != HDC_CODEC_SCHEMA_VERSION {
            return Err("unsupported HDC codec schema version".into());
        }
        if self.dimension != BinaryHV::DIM || self.bits.len() != BinaryHV::BYTES {
            return Err(format!(
                "invalid HDC binary frame dimensions: expected {} bits / {} bytes, got {} / {}",
                BinaryHV::DIM,
                BinaryHV::BYTES,
                self.dimension,
                self.bits.len()
            ));
        }

        let bytes: [u8; BinaryHV::BYTES] = self
            .bits
            .clone()
            .try_into()
            .map_err(|_| "HDC binary frame byte length mismatch".to_string())?;

        Ok(BinaryHV(bytes))
    }
}

pub fn quantize_continuous(
    continuous: &ContinuousHV,
) -> Result<(BinaryHV, HdcQuantizationMetrics), String> {
    validate_continuous(continuous)?;
    let binary = BinaryHV::from_bipolar(&continuous.values);
    let reconstructed = binary.to_continuous();

    let sign_disagreement_rate = continuous
        .values
        .iter()
        .zip(&reconstructed.values)
        .filter(|(original, quantized)| {
            ((**original > 0.0) != (**quantized > 0.0))
                || (**original == 0.0 && **quantized > 0.0)
        })
        .count() as f64
        / continuous.values.len() as f64;

    let cosine_similarity = continuous.similarity(&reconstructed) as f64;
    let continuous_bytes = continuous.values.len() * std::mem::size_of::<f32>();
    let binary_bytes = BinaryHV::BYTES;
    let compression_ratio = continuous_bytes as f64 / binary_bytes as f64;

    let metrics = HdcQuantizationMetrics {
        schema_version: HDC_CODEC_SCHEMA_VERSION,
        dimension: continuous.values.len(),
        continuous_bytes,
        binary_bytes,
        compression_ratio,
        cosine_similarity,
        sign_disagreement_rate,
    };

    if !metrics.validates() {
        return Err("invalid HDC quantization metrics".into());
    }

    Ok((binary, metrics))
}

pub fn roundtrip_continuous(continuous: &ContinuousHV) -> Result<ContinuousHV, String> {
    let (binary, _) = quantize_continuous(continuous)?;
    Ok(binary.to_continuous())
}

fn validate_continuous(continuous: &ContinuousHV) -> Result<(), String> {
    if continuous.values.len() != HDC_DIMENSION {
        return Err(format!(
            "invalid continuous HDC dimension: expected {}, got {}",
            HDC_DIMENSION,
            continuous.values.len()
        ));
    }

    if continuous.values.iter().any(|value| !value.is_finite()) {
        return Err("continuous HDC vector contains non-finite values".into());
    }

    let norm = continuous.norm();
    if !norm.is_finite() || norm <= 1e-10 {
        return Err("continuous HDC vector must have a finite non-zero norm".into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_quantization_is_reproducible() {
        let continuous = ContinuousHV::random(HDC_DIMENSION, 42);
        let (a, metrics_a) = quantize_continuous(&continuous).unwrap();
        let (b, metrics_b) = quantize_continuous(&continuous).unwrap();
        assert_eq!(a, b);
        assert_eq!(metrics_a, metrics_b);
    }

    #[test]
    fn binary_frame_roundtrips_exactly() {
        let continuous = ContinuousHV::random(HDC_DIMENSION, 7);
        let (binary, _) = quantize_continuous(&continuous).unwrap();
        let frame = HdcBinaryFrame::from_binary(&binary);
        let restored = frame.to_binary().unwrap();
        assert_eq!(binary, restored);
    }

    #[test]
    fn quantization_metrics_are_bounded() {
        let continuous = ContinuousHV::random(HDC_DIMENSION, 99);
        let (_, metrics) = quantize_continuous(&continuous).unwrap();
        assert!(metrics.validates());
        assert!(metrics.compression_ratio > 1.0);
        assert!((0.0..=1.0).contains(&metrics.sign_disagreement_rate));
    }

    #[test]
    fn nonfinite_and_wrong_dimension_inputs_are_rejected() {
        let wrong_dim = ContinuousHV::from_vec(vec![1.0; 32]);
        assert!(quantize_continuous(&wrong_dim).is_err());

        let mut nonfinite = ContinuousHV::random(HDC_DIMENSION, 11);
        nonfinite.values[3] = f32::NAN;
        assert!(quantize_continuous(&nonfinite).is_err());
    }

    #[test]
    fn zero_vector_is_rejected() {
        let zero = ContinuousHV::zero(HDC_DIMENSION);
        assert!(quantize_continuous(&zero).is_err());
    }
}
