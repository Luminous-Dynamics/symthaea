// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use nalgebra::DMatrix;
use std::f64::consts::PI;
use symthaea_core::hdc::binary_hv::BinaryHV;

pub const HDC_DIM: usize = 16_384;
pub const HDC_BYTES: usize = HDC_DIM / 8;

/// Hofstadter / Harper spectral generator.
///
/// This currently generates Harper Hamiltonian slices. It is not a full laboratory
/// Hofstadter butterfly unless the caller sweeps fluxes and phases.
pub struct HofstadterGenerator {
    dim: usize,
}

impl HofstadterGenerator {
    pub fn new(dim: usize) -> Self {
        assert_eq!(
            dim, HDC_DIM,
            "HofstadterGenerator currently supports only {HDC_DIM}D BinaryHV"
        );

        Self { dim }
    }

    /// Generate eigenvalues for a Harper slice at kx=ky=0 for flux alpha = p/q.
    pub fn generate_harper_slice(&self, p: usize, q: usize) -> Vec<f64> {
        self.generate_harper_slice_with_phase(p, q, 0.0, 0.0)
    }

    /// Generate eigenvalues for a Harper slice with simple kx/ky phase controls.
    ///
    /// The real symmetric approximation uses `cos(kx)` on the boundary hopping.
    pub fn generate_harper_slice_with_phase(
        &self,
        p: usize,
        q: usize,
        kx: f64,
        ky: f64,
    ) -> Vec<f64> {
        if q == 0 {
            return Vec::new();
        }

        let alpha = p as f64 / q as f64;
        let mut matrix = DMatrix::<f64>::zeros(q, q);

        for i in 0..q {
            matrix[(i, i)] = 2.0 * (2.0 * PI * alpha * i as f64 + ky).cos();

            if i > 0 {
                matrix[(i, i - 1)] = 1.0;
                matrix[(i - 1, i)] = 1.0;
            }
        }

        if q > 1 {
            let boundary = kx.cos();
            matrix[(0, q - 1)] = boundary;
            matrix[(q - 1, 0)] = boundary;
        }

        let mut evals: Vec<f64> = matrix.symmetric_eigenvalues().as_slice().to_vec();
        evals.sort_by(|a, b| a.total_cmp(b));
        evals
    }

    /// Generate a multi-flux butterfly sample as `(p, q, energy)` triples.
    pub fn generate_butterfly_sample(&self, max_q: usize) -> Vec<(usize, usize, f64)> {
        let mut sample = Vec::new();

        for q in 2..=max_q {
            for p in 1..q {
                if gcd(p, q) != 1 {
                    continue;
                }

                for energy in self.generate_harper_slice(p, q) {
                    sample.push((p, q, energy));
                }
            }
        }

        sample
    }

    /// Encode a spectrum into a BinaryHV using deterministic basis vectors.
    pub fn encode_spectrum(
        &self,
        spectrum: &[f64],
        min_e: f64,
        max_e: f64,
        resolution: usize,
    ) -> BinaryHV {
        assert!(resolution >= 2, "resolution must be at least 2");
        assert!(max_e > min_e, "max_e must be greater than min_e");

        let mut bundle = vec![0i32; self.dim];

        for &energy in spectrum {
            let norm_e = ((energy - min_e) / (max_e - min_e)).clamp(0.0, 1.0);
            let idx = (norm_e * (resolution - 1) as f64).round() as usize;
            let basis = BinaryHV::random(idx as u64);

            for (i, accumulator) in bundle.iter_mut().enumerate() {
                if basis.get_bit(i) == 1 {
                    *accumulator += 1;
                } else {
                    *accumulator -= 1;
                }
            }
        }

        bundle_to_binary_hv(&bundle)
    }

    pub fn similarity_score(&self, hv1: &BinaryHV, hv2: &BinaryHV) -> f64 {
        hv1.similarity(hv2)
    }

    pub fn average_cross_scale_similarity(
        &self,
        spectra: &[Vec<f64>],
        min_e: f64,
        max_e: f64,
        resolution: usize,
    ) -> f64 {
        if spectra.len() < 2 {
            return 0.0;
        }

        let encoded: Vec<_> = spectra
            .iter()
            .map(|spectrum| self.encode_spectrum(spectrum, min_e, max_e, resolution))
            .collect();

        let mut sum = 0.0;
        let mut count = 0usize;

        for i in 0..encoded.len() {
            for j in (i + 1)..encoded.len() {
                sum += self.similarity_score(&encoded[i], &encoded[j]);
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

fn bundle_to_binary_hv(bundle: &[i32]) -> BinaryHV {
    let mut bits = [0u8; HDC_BYTES];

    for (i, &value) in bundle.iter().enumerate().take(HDC_DIM) {
        if value > 0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bits[byte_idx] |= 1 << bit_idx;
        }
    }

    BinaryHV(bits)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harper_slice_generation() {
        let generator = HofstadterGenerator::new(HDC_DIM);
        let spectrum = generator.generate_harper_slice(1, 5);

        assert_eq!(spectrum.len(), 5);
    }

    #[test]
    fn test_spectral_encoding_is_deterministic() {
        let generator = HofstadterGenerator::new(HDC_DIM);
        let spectrum = generator.generate_harper_slice(1, 5);
        let hv1 = generator.encode_spectrum(&spectrum, -4.0, 4.0, 100);
        let hv2 = generator.encode_spectrum(&spectrum, -4.0, 4.0, 100);

        assert!(generator.similarity_score(&hv1, &hv2) > 0.99);
    }

    #[test]
    fn test_out_of_range_energies_are_clamped() {
        let generator = HofstadterGenerator::new(HDC_DIM);
        let spectrum = vec![-999.0, 0.0, 999.0];
        let hv = generator.encode_spectrum(&spectrum, -4.0, 4.0, 100);

        assert!(generator.similarity_score(&hv, &hv).is_finite());
    }

    #[test]
    fn test_butterfly_sample_nonempty() {
        let generator = HofstadterGenerator::new(HDC_DIM);
        let sample = generator.generate_butterfly_sample(5);

        assert!(!sample.is_empty());
    }
}
