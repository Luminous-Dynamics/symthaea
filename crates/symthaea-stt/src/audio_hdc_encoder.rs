// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio HDC Encoder — mel spectrogram frames → 16,384D ContinuousHV
//!
//! Encodes each mel frame as a weighted bundle of genesis-seeded basis vectors,
//! projecting 40D mel features into the full HDC space where the unified
//! HdcLtcNeuron operates.
//!
//! This replaces the information-destroying BLAKE3 random projection with
//! a structured, invertible encoding that preserves spectral relationships.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Encodes mel spectrogram frames to 16,384D hypervectors.
///
/// Each mel frequency bin gets a deterministic basis HV from the genesis seed.
/// The frame's HV is a weighted bundle: Σ(mel[i] × basis[i]), normalized.
pub struct AudioHdcEncoder {
    /// One basis HV per mel bin (genesis-seeded, quasi-orthogonal)
    mel_basis: Vec<ContinuousHV>,
    /// Number of mel bins
    mel_dim: usize,
}

impl AudioHdcEncoder {
    /// Create a new encoder with genesis-seeded basis vectors.
    pub fn new(mel_dim: usize, genesis: &GenesisSeed) -> Self {
        let mel_basis = (0..mel_dim)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("audio_hdc_encoder::mel_bin_{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        Self { mel_basis, mel_dim }
    }

    /// Encode a single mel spectrogram frame to 16,384D.
    ///
    /// Uses weighted bundling: HV = normalize(Σ mel[i] × basis[i])
    /// This preserves spectral structure — similar spectra produce similar HVs.
    pub fn encode_frame(&self, mel: &[f32]) -> ContinuousHV {
        let mut result = vec![0.0f32; HDC_DIMENSION];

        // Weighted sum of basis vectors
        let mut total_weight = 0.0f32;
        for (i, &val) in mel.iter().take(self.mel_dim).enumerate() {
            if val.abs() > 1e-8 {
                let weight = val.abs();
                total_weight += weight;
                let sign = val.signum();
                for (j, &basis_val) in self.mel_basis[i].values.iter().enumerate() {
                    result[j] += sign * weight * basis_val;
                }
            }
        }

        // Normalize to unit length
        if total_weight > 1e-8 {
            let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut result {
                *x /= norm;
            }
        }

        ContinuousHV::from_values(result)
    }

    /// Mel dimension.
    pub fn mel_dim(&self) -> usize {
        self.mel_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_valid_hv() {
        let genesis = GenesisSeed::from_phrase("test_audio_encoder");
        let encoder = AudioHdcEncoder::new(40, &genesis);

        let mel = vec![0.5f32; 40];
        let hv = encoder.encode_frame(&mel);

        assert_eq!(hv.dim(), HDC_DIMENSION);
        // Should be approximately unit length
        let norm = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "HV should be unit length: {}",
            norm
        );
    }

    #[test]
    fn test_similar_spectra_produce_similar_hvs() {
        let genesis = GenesisSeed::from_phrase("test_similarity");
        let encoder = AudioHdcEncoder::new(40, &genesis);

        let mel_a = vec![0.5f32; 40];
        let mut mel_b = mel_a.clone();
        mel_b[0] += 0.1; // Slightly different

        let hv_a = encoder.encode_frame(&mel_a);
        let hv_b = encoder.encode_frame(&mel_b);

        let similarity = hv_a.similarity(&hv_b);
        assert!(
            similarity > 0.9,
            "Similar spectra should produce similar HVs: sim={}",
            similarity
        );
    }

    #[test]
    fn test_different_spectra_produce_different_hvs() {
        let genesis = GenesisSeed::from_phrase("test_different");
        let encoder = AudioHdcEncoder::new(40, &genesis);

        let mut mel_low = vec![0.0f32; 40];
        mel_low[0..10].fill(1.0); // Low frequencies

        let mut mel_high = vec![0.0f32; 40];
        mel_high[30..40].fill(1.0); // High frequencies

        let hv_low = encoder.encode_frame(&mel_low);
        let hv_high = encoder.encode_frame(&mel_high);

        let similarity = hv_low.similarity(&hv_high);
        assert!(
            similarity < 0.5,
            "Different spectra should produce different HVs: sim={}",
            similarity
        );
    }

    #[test]
    fn test_zero_frame_produces_zero_hv() {
        let genesis = GenesisSeed::from_phrase("test_zero");
        let encoder = AudioHdcEncoder::new(40, &genesis);

        let mel = vec![0.0f32; 40];
        let hv = encoder.encode_frame(&mel);

        let norm = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm < 0.01,
            "Zero mel should produce near-zero HV: norm={}",
            norm
        );
    }
}
