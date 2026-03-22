// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hyperdimensional Computing (HDC) primitives for audio pattern encoding.
//!
//! This module provides the core HDC building blocks:
//! - `HV`: Hypervector operations (add, bind, permute, similarity)
//! - `SparseProjector`: Discrete quantization + binary codebook (Grey Goo fix)
//! - `RffProjector`: Random Fourier Feature projection (topology-preserving)

use std::f32::consts::PI;

/// Standard HDC dimension (2^14 for high capacity)
pub const HDC_DIM: usize = 16384;

/// Number of quantization bins for sparse projection
pub const SPARSE_BINS: usize = 16;

/// Default bandwidth parameter for RFF kernel
pub const RFF_GAMMA: f32 = 0.5;

/// Adaptive gamma values for different feature types
pub const RFF_GAMMA_TIMBRE: f32 = 0.7;
pub const RFF_GAMMA_RHYTHM: f32 = 0.4;

// =============================================================================
// Hypervector (HV)
// =============================================================================

/// Hypervector - the core representation in HDC
#[derive(Clone)]
pub struct HV {
    pub values: Vec<f32>,
}

impl HV {
    pub fn zero() -> Self {
        Self {
            values: vec![0.0; HDC_DIM],
        }
    }

    pub fn random_seeded(seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut values = vec![0.0; HDC_DIM];
        for i in 0..HDC_DIM {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            values[i] = ((h as f32 / u64::MAX as f32) - 0.5) * 2.0;
        }
        Self { values }
    }

    pub fn from_scalar(value: f32, basis: &HV) -> Self {
        basis.scale(value)
    }

    pub fn similarity(&self, other: &HV) -> f32 {
        let dot: f32 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a > 1e-6 && mag_b > 1e-6 {
            dot / (mag_a * mag_b)
        } else {
            0.0
        }
    }

    pub fn add(&self, other: &HV) -> HV {
        HV {
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn scale(&self, s: f32) -> HV {
        HV {
            values: self.values.iter().map(|v| v * s).collect(),
        }
    }

    /// Binding operation (element-wise multiply) - creates association
    pub fn bind(&self, other: &HV) -> HV {
        HV {
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(a, b)| a * b)
                .collect(),
        }
    }

    /// Permutation (circular shift) - creates sequence/order
    pub fn permute(&self, shift: usize) -> HV {
        let mut values = vec![0.0; HDC_DIM];
        for i in 0..HDC_DIM {
            values[(i + shift) % HDC_DIM] = self.values[i];
        }
        HV { values }
    }

    pub fn normalize(&self) -> HV {
        let mag: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 1e-6 {
            self.scale(1.0 / mag)
        } else {
            self.clone()
        }
    }

    /// Create a BINARY random vector (+1 or -1 only)
    pub fn random_binary_seeded(seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut values = vec![0.0; HDC_DIM];
        for i in 0..HDC_DIM {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            values[i] = if h & 1 == 0 { 1.0 } else { -1.0 };
        }
        Self { values }
    }

    /// XOR-like binding for binary vectors (element-wise multiply)
    pub fn xor_bind(&self, other: &HV) -> HV {
        HV {
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(a, b)| a * b)
                .collect(),
        }
    }
}

// =============================================================================
// Sparse Binary Projection - The Grey Goo Fix
// =============================================================================

/// Sparse Binary Projector - Maps continuous features to discrete HDC space
pub struct SparseProjector {
    num_features: usize,
    codebook: Vec<HV>,
}

impl SparseProjector {
    /// Create a new sparse projector with random binary codebook
    pub fn new(num_features: usize, seed_base: u64) -> Self {
        let codebook_size = num_features * SPARSE_BINS;
        let mut codebook = Vec::with_capacity(codebook_size);

        for i in 0..codebook_size {
            codebook.push(HV::random_binary_seeded(seed_base + i as u64));
        }

        Self {
            num_features,
            codebook,
        }
    }

    /// Project continuous features `[0.0, 1.0]` to a sparse binary HDC vector
    pub fn project(&self, features: &[f32]) -> HV {
        let mut result = HV::zero();

        for (i, &value) in features.iter().enumerate() {
            if i >= self.num_features {
                break;
            }

            let clamped = value.clamp(0.0, 0.9999);
            let bin = (clamped * SPARSE_BINS as f32) as usize;
            let codebook_idx = i * SPARSE_BINS + bin;

            result = result.add(&self.codebook[codebook_idx]);
        }

        result.normalize()
    }

    /// Project with activity masking - only include bins above a threshold
    pub fn project_masked(&self, features: &[f32], threshold: f32) -> HV {
        let mut result = HV::zero();
        let mut active_count = 0;

        for (i, &value) in features.iter().enumerate() {
            if i >= self.num_features {
                break;
            }

            if value < threshold {
                continue;
            }

            let clamped = value.clamp(0.0, 0.9999);
            let bin = (clamped * SPARSE_BINS as f32) as usize;
            let codebook_idx = i * SPARSE_BINS + bin;

            result = result.add(&self.codebook[codebook_idx]);
            active_count += 1;
        }

        if active_count > 0 {
            result.normalize()
        } else {
            result
        }
    }
}

// =============================================================================
// Random Fourier Feature (RFF) Projector - Topology-Preserving Encoding
// =============================================================================

/// Random Fourier Feature Projector
/// Maps continuous vectors to binary HDC vectors while preserving Euclidean topology.
pub struct RffProjector {
    input_dim: usize,
    output_dim: usize,
    #[allow(dead_code)]
    gamma: f32,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl RffProjector {
    /// Create a new RFF projector with default gamma
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        Self::with_gamma(input_dim, output_dim, seed, RFF_GAMMA)
    }

    /// Create a new RFF projector with custom gamma value
    pub fn with_gamma(input_dim: usize, output_dim: usize, seed: u64, gamma: f32) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut weights = Vec::with_capacity(output_dim * input_dim);
        let mut biases = Vec::with_capacity(output_dim);

        // Initialize weights with Gaussian distribution N(0, gamma^2)
        for i in 0..(output_dim * input_dim) {
            let mut hasher1 = DefaultHasher::new();
            let mut hasher2 = DefaultHasher::new();
            seed.hash(&mut hasher1);
            (i * 2).hash(&mut hasher1);
            seed.hash(&mut hasher2);
            (i * 2 + 1).hash(&mut hasher2);

            let u1 = (hasher1.finish() as f64 / u64::MAX as f64).max(1e-10);
            let u2 = hasher2.finish() as f64 / u64::MAX as f64;

            // Box-Muller transform
            let gaussian =
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
            weights.push(gaussian * gamma);
        }

        // Initialize biases with Uniform [0, 2pi]
        for i in 0..output_dim {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            (i + 1_000_000).hash(&mut hasher);
            let u = hasher.finish() as f32 / u64::MAX as f32;
            biases.push(u * 2.0 * PI);
        }

        Self {
            input_dim,
            output_dim,
            gamma,
            weights,
            biases,
        }
    }

    /// Project a continuous state vector into HDC space
    pub fn project(&self, input: &[f32]) -> HV {
        assert!(
            input.len() <= self.input_dim,
            "Input dimension {} exceeds projector dimension {}",
            input.len(),
            self.input_dim
        );

        let mut values = vec![0.0f32; HDC_DIM];

        for i in 0..self.output_dim.min(HDC_DIM) {
            let mut activation = 0.0f32;
            for j in 0..input.len() {
                activation += input[j] * self.weights[i * self.input_dim + j];
            }

            let z = (activation + self.biases[i]).cos();
            values[i] = if z > 0.0 { 1.0 } else { -1.0 };
        }

        // Fill remaining dimensions with pattern if output_dim < HDC_DIM
        if self.output_dim < HDC_DIM {
            for i in self.output_dim..HDC_DIM {
                values[i] = values[i % self.output_dim];
            }
        }

        HV { values }
    }

    /// Project with L2 normalization of input (amplitude-invariant)
    pub fn project_normalized(&self, input: &[f32]) -> HV {
        let mag: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 1e-6 {
            let normalized: Vec<f32> = input.iter().map(|x| x / mag).collect();
            self.project(&normalized)
        } else {
            self.project(input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdc_binding() {
        let a = HV::random_seeded(1);
        let b = HV::random_seeded(2);
        let bound = a.bind(&b);

        // Binding creates quasi-orthogonal vectors (dissimilar to originals)
        let sim_to_a = bound.similarity(&a);
        let sim_to_b = bound.similarity(&b);
        // Bound vector should not be too similar to originals
        assert!(
            sim_to_a.abs() < 0.5,
            "Bound vector should be dissimilar to a, got {}",
            sim_to_a
        );
        assert!(
            sim_to_b.abs() < 0.5,
            "Bound vector should be dissimilar to b, got {}",
            sim_to_b
        );
    }

    #[test]
    fn test_sparse_projector() {
        let projector = SparseProjector::new(10, 1000);
        let features = vec![0.5; 10];
        let hv = projector.project(&features);

        // Should produce a normalized vector
        let mag: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 0.01, "Vector should be normalized");
    }
}
