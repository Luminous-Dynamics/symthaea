// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC vector compression via TurboQuant (PolarQuant + QJL).
//!
//! Compresses 16,384D ContinuousHV vectors from 64KB to ~10KB using
//! PolarQuant's optimal scalar quantization in polar coordinates.
//!
//! # Integration Points
//!
//! - **Episodic memory** (`memory_bridge.rs`): Compressed full HDC vectors attached
//!   to memories via `EpisodicMemoryBridge::attach_compressed_hv()`. The parallel
//!   learning path (`helpers/parallel.rs`) compresses the full HDV when encoding
//!   high-Phi experiences.
//!
//! - **Quarantined shared-mask wisdom demo**: NOT compressed — uses BinaryHV
//!   (2048 bytes = 16,384 bits), which is already 32x smaller than ContinuousHV.
//!   PolarQuant operates on f32 vectors and cannot improve on 1-bit encoding.
//!
//! - **Dream consolidation**: NOT compressed — dream replay uses phi/surprise scores
//!   and BinaryHV bindings, not raw ContinuousHV storage.
//!
//! - **Mesh wisdom packets**: NOT compressed — WisdomPacket carries BinaryHV (2048
//!   bytes) in a fixed 2,104-byte version-2 frame. Already optimally compact.
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::hv_compression::{HvCompressor, CompressedHv};
//!
//! let compressor = HvCompressor::new(4).unwrap(); // 4-bit quantization
//! let compressed = compressor.compress(&my_hv);
//! let reconstructed = compressor.decompress(&compressed);
//! assert!(compressed.compression_ratio() > 3.0);
//! ```

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use turbo_quant::polar::{PolarCode, PolarQuantizer};

/// Compressed representation of a ContinuousHV.
///
/// Holds the PolarQuant-encoded data plus metadata for reconstruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedHv {
    /// PolarQuant-encoded code for the whole vector.
    code: PolarCode,
    /// Original dimensionality.
    dim: usize,
    /// Bits per dimension used for quantization.
    bits: u8,
}

impl CompressedHv {
    /// Compression ratio: original bytes / estimated compressed bytes.
    ///
    /// Original: `dim * 4` bytes (f32). Compressed: ~`dim * bits / 8` bytes.
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dim * 4; // f32 = 4 bytes
        let compressed_bytes = (self.dim * self.bits as usize + 7) / 8;
        if compressed_bytes == 0 {
            return f32::INFINITY;
        }
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Original dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bits per dimension.
    pub fn bits(&self) -> u8 {
        self.bits
    }
}

/// HDC vector compressor using PolarQuant.
///
/// Wraps a `PolarQuantizer` configured for HDC_DIMENSION-sized vectors.
/// Thread-safe (`Send + Sync`) — the quantizer is immutable after construction.
pub struct HvCompressor {
    quantizer: PolarQuantizer,
    bits: u8,
}

impl HvCompressor {
    /// Create a new compressor with the given bit-width (1-16).
    ///
    /// - 4 bits: ~8x compression, good cosine similarity preservation (>0.95)
    /// - 8 bits: ~4x compression, near-lossless
    ///
    /// Uses a fixed seed (42) for reproducible rotations.
    pub fn new(bits: u8) -> Result<Self, turbo_quant::error::TurboQuantError> {
        let quantizer = PolarQuantizer::new(HDC_DIMENSION, bits, 42)?;
        Ok(Self { quantizer, bits })
    }

    /// Create a compressor for an arbitrary dimension.
    pub fn with_dim(dim: usize, bits: u8) -> Result<Self, turbo_quant::error::TurboQuantError> {
        let quantizer = PolarQuantizer::new(dim, bits, 42)?;
        Ok(Self { quantizer, bits })
    }

    /// Compress a ContinuousHV into a CompressedHv.
    pub fn compress(&self, hv: &ContinuousHV) -> CompressedHv {
        let values = &hv.values;
        let code = self
            .quantizer
            .encode(values)
            .expect("PolarQuantizer::encode failed — dimension mismatch");
        CompressedHv {
            code,
            dim: values.len(),
            bits: self.bits,
        }
    }

    /// Compress a raw f32 slice.
    pub fn compress_slice(&self, values: &[f32]) -> CompressedHv {
        let code = self
            .quantizer
            .encode(values)
            .expect("PolarQuantizer::encode failed — dimension mismatch");
        CompressedHv {
            code,
            dim: values.len(),
            bits: self.bits,
        }
    }

    /// Decompress back to a ContinuousHV.
    pub fn decompress(&self, compressed: &CompressedHv) -> ContinuousHV {
        let values = self
            .quantizer
            .decode(&compressed.code)
            .expect("PolarQuantizer::decode failed");
        ContinuousHV::from_vec(values)
    }

    /// Decompress to a raw Vec<f32>.
    pub fn decompress_to_vec(&self, compressed: &CompressedHv) -> Vec<f32> {
        self.quantizer
            .decode(&compressed.code)
            .expect("PolarQuantizer::decode failed")
    }

    /// Bits per dimension.
    pub fn bits(&self) -> u8 {
        self.bits
    }
}

/// Cosine similarity between two f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_cosine_similarity() {
        let compressor = HvCompressor::new(4).unwrap();
        // Create a non-trivial vector
        let mut values = vec![0.0f32; HDC_DIMENSION];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as f32) * 0.01).sin();
        }
        let hv = ContinuousHV::from_vec(values.clone());
        let compressed = compressor.compress(&hv);
        let reconstructed = compressor.decompress(&compressed);
        let sim = cosine_similarity(&values, &reconstructed.values);
        assert!(
            sim > 0.90,
            "Cosine similarity after roundtrip should be > 0.90, got {sim}"
        );
    }

    #[test]
    fn compression_ratio_exceeds_3x() {
        let compressor = HvCompressor::new(4).unwrap();
        let hv = ContinuousHV::random_default(42);
        let compressed = compressor.compress(&hv);
        let ratio = compressed.compression_ratio();
        assert!(
            ratio > 3.0,
            "Compression ratio should exceed 3x, got {ratio}"
        );
    }

    #[test]
    fn zero_vector_compresses() {
        let compressor = HvCompressor::new(4).unwrap();
        let hv = ContinuousHV::from_vec(vec![0.0; HDC_DIMENSION]);
        let compressed = compressor.compress(&hv);
        let reconstructed = compressor.decompress(&compressed);
        // Zero vector should decompress to near-zero
        let max_abs: f32 = reconstructed
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1.0,
            "Decompressed zero vector should have small values, got max {max_abs}"
        );
    }

    #[test]
    fn random_vectors_compress() {
        let compressor = HvCompressor::new(4).unwrap();
        for seed in 0..5u64 {
            let hv = ContinuousHV::random_default(seed);
            let compressed = compressor.compress(&hv);
            let _reconstructed = compressor.decompress(&compressed);
            assert_eq!(compressed.dim(), HDC_DIMENSION);
            assert_eq!(compressed.bits(), 4);
        }
    }

    #[test]
    fn eight_bit_higher_fidelity() {
        let compressor_4 = HvCompressor::new(4).unwrap();
        let compressor_8 = HvCompressor::new(8).unwrap();
        let hv = ContinuousHV::random_default(99);
        let original = hv.values.clone();

        let reconstructed_4 = compressor_4.decompress(&compressor_4.compress(&hv));
        let reconstructed_8 = compressor_8.decompress(&compressor_8.compress(&hv));

        let sim_4 = cosine_similarity(&original, &reconstructed_4.values);
        let sim_8 = cosine_similarity(&original, &reconstructed_8.values);
        assert!(
            sim_8 >= sim_4,
            "8-bit ({sim_8}) should be >= 4-bit ({sim_4}) fidelity"
        );
    }

    #[test]
    fn compressed_hv_serializes() {
        let compressor = HvCompressor::new(4).unwrap();
        let hv = ContinuousHV::random_default(7);
        let compressed = compressor.compress(&hv);
        let json = serde_json::to_string(&compressed).expect("serialize");
        let deserialized: CompressedHv = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.dim(), compressed.dim());
        assert_eq!(deserialized.bits(), compressed.bits());
    }
}
