use serde::{Deserialize, Serialize};

/// Default HyperFeel hypervector.
///
/// This is a simple, backend-agnostic representation intended
/// to be efficient enough for prototypes while remaining easy
/// to swap out for Symthaea's HV16 implementation in the future.
///
/// - Dimension is fixed to 16,384 by convention (see `DIMENSION`).
/// - Elements are signed 8-bit integers after quantization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hypervector {
    /// Quantized components in [-128, 127].
    pub data: Vec<i8>,
}

impl Hypervector {
    /// Conventional dimension for HyperFeel hypervectors.
    pub const DIMENSION: usize = 16_384;

    /// Create an all-zero hypervector.
    pub fn zero() -> Self {
        Self {
            data: vec![0; Self::DIMENSION],
        }
    }

    /// Element-wise addition with saturation.
    pub fn add(&self, other: &Self) -> Self {
        let mut out = Vec::with_capacity(Self::DIMENSION);
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let sum = (*a as i16) + (*b as i16);
            let clamped = sum.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
            out.push(clamped);
        }
        Self { data: out }
    }

    /// Normalize by dividing by a scalar and re-quantizing.
    pub fn normalize(&self, count: usize) -> Self {
        if count == 0 {
            return Self::zero();
        }
        let factor = 1.0f32 / (count as f32);
        let mut out = Vec::with_capacity(Self::DIMENSION);
        for &v in &self.data {
            let scaled = (v as f32) * factor;
            let rounded = scaled.round().clamp(i8::MIN as f32, i8::MAX as f32);
            out.push(rounded as i8);
        }
        Self { data: out }
    }

    /// Cosine-like similarity in [-1.0, 1.0].
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut dot: i64 = 0;
        let mut norm_a: i64 = 0;
        let mut norm_b: i64 = 0;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let a_i = *a as i64;
            let b_i = *b as i64;
            dot += a_i * b_i;
            norm_a += a_i * a_i;
            norm_b += b_i * b_i;
        }
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }
        let denom = ((norm_a as f64).sqrt() * (norm_b as f64).sqrt()) as f32;
        (dot as f32 / denom).clamp(-1.0, 1.0)
    }
}

