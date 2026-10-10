//! Explicit continuous/binary HDC codec boundary.
//!
//! This module is opt-in because the communication crate should remain
//! lightweight unless a deployment explicitly wants Symthaea HDC integration.
//! The codec measures quantization separately from semantic decoding.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const HDC_CODEC_SCHEMA_VERSION: u16 = 1;
/// Maximum raw JSON byte length accepted by bounded HDC codec artifact parsers.
pub const HDC_CODEC_MAX_SERIALIZED_ARTIFACT_BYTES: usize = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdcCodecDescriptor {
    pub schema_version: u16,
    pub codec_id: String,
    pub source_representation: String,
    pub target_representation: String,
    pub quantizer: String,
    pub dimension: usize,
    pub encoder_revision: Option<String>,
    pub codebook_hash: Option<String>,
}

impl HdcCodecDescriptor {
    pub fn v1() -> Self {
        Self {
            schema_version: HDC_CODEC_SCHEMA_VERSION,
            codec_id: "symthaea.hdc.continuous-sign-v1".into(),
            source_representation: "ContinuousHV:f32".into(),
            target_representation: "BinaryHV:bits".into(),
            quantizer: "sign(value > 0)".into(),
            dimension: HDC_DIMENSION,
            encoder_revision: None,
            codebook_hash: None,
        }
    }

    pub fn validates(&self) -> bool {
        self.schema_version == HDC_CODEC_SCHEMA_VERSION
            && self.codec_id == "symthaea.hdc.continuous-sign-v1"
            && self.source_representation == "ContinuousHV:f32"
            && self.target_representation == "BinaryHV:bits"
            && self.quantizer == "sign(value > 0)"
            && self.dimension == HDC_DIMENSION
    }
}

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

    /// Deserialize a binary frame only after enforcing the raw byte-size
    /// ceiling and the exact frame shape.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > HDC_CODEC_MAX_SERIALIZED_ARTIFACT_BYTES {
            return Err(format!(
                "HDC binary frame JSON exceeds {} bytes",
                HDC_CODEC_MAX_SERIALIZED_ARTIFACT_BYTES
            ));
        }
        let frame: Self =
            serde_json::from_slice(bytes).map_err(|error| format!("HDC binary frame JSON: {error}"))?;
        frame.to_binary()?;
        Ok(frame)
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HdcBitCorruptionObservation {
    pub flip_probability: f32,
    pub seed: u64,
    pub binary_similarity: f64,
    pub cosine_similarity_to_original: f64,
}

pub fn measure_bit_corruption(
    continuous: &ContinuousHV,
    flip_probability: f32,
    seed: u64,
) -> Result<HdcBitCorruptionObservation, String> {
    let (binary, _) = quantize_continuous(continuous)?;
    if !flip_probability.is_finite() || !(0.0..=1.0).contains(&flip_probability) {
        return Err("flip probability must be finite and within [0, 1]".into());
    }

    let corrupted = binary.add_noise(flip_probability, seed);
    let restored = corrupted.to_continuous();
    let binary_similarity = binary.similarity(&corrupted) as f64;
    let cosine_similarity_to_original = full_cosine_similarity(continuous, &restored);

    Ok(HdcBitCorruptionObservation {
        flip_probability,
        seed,
        binary_similarity,
        cosine_similarity_to_original,
    })
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
        .filter(|(original, quantized)| (**original > 0.0) != (**quantized > 0.0))
        .count() as f64
        / continuous.values.len() as f64;

    let cosine_similarity = full_cosine_similarity(continuous, &reconstructed);
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

fn full_cosine_similarity(a: &ContinuousHV, b: &ContinuousHV) -> f64 {
    let mut dot = 0.0_f64;
    let mut norm_a_sq = 0.0_f64;
    let mut norm_b_sq = 0.0_f64;

    for (x, y) in a.values.iter().zip(&b.values) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let denominator = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
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
    fn codec_descriptor_is_stable_and_valid() {
        let descriptor = HdcCodecDescriptor::v1();
        assert!(descriptor.validates());
        assert_eq!(descriptor.codec_id, "symthaea.hdc.continuous-sign-v1");
    }

    #[test]
    fn bit_corruption_measurement_is_deterministic() {
        let continuous = ContinuousHV::random(HDC_DIMENSION, 123);
        let a = measure_bit_corruption(&continuous, 0.01, 77).unwrap();
        let b = measure_bit_corruption(&continuous, 0.01, 77).unwrap();
        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a.binary_similarity));
        assert!((-1.0..=1.0).contains(&a.cosine_similarity_to_original));
    }

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

        let encoded = serde_json::to_vec(&frame).unwrap();
        let parsed = HdcBinaryFrame::from_json_bytes(&encoded).unwrap();
        assert_eq!(parsed, frame);

        let oversized = vec![b' '; HDC_CODEC_MAX_SERIALIZED_ARTIFACT_BYTES + 1];
        assert!(HdcBinaryFrame::from_json_bytes(&oversized).is_err());

        let mut malformed = frame.clone();
        malformed.bits.pop();
        let malformed_json = serde_json::to_vec(&malformed).unwrap();
        assert!(HdcBinaryFrame::from_json_bytes(&malformed_json).is_err());
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
