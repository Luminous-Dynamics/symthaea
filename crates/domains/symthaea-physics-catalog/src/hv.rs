// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight hypervector implementation for WASM.
//!
//! Self-contained f32 vector with cosine similarity — no external dependencies.
//! Uses 1024 dimensions (reduced from 16,384) for WASM binary size.

use serde::{Deserialize, Serialize};

/// Hypervector dimension for the WASM catalog (reduced for binary size).
pub const DIM: usize = 1024;

/// A lightweight continuous hypervector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HV {
    pub values: Vec<f32>,
}

impl HV {
    /// Create a zero vector.
    pub fn zero() -> Self {
        Self {
            values: vec![0.0; DIM],
        }
    }

    /// Create a deterministic pseudo-random vector from a seed.
    ///
    /// Uses a simple xorshift PRNG seeded from the input — deterministic,
    /// no external dependencies, WASM-compatible.
    pub fn from_seed(seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let values = (0..DIM)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Convert to f32 in [-1, 1]
                let bits = (state >> 40) as u32;
                (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        Self { values }
    }

    /// Create from a string hash (deterministic name encoding).
    pub fn from_name(name: &str) -> Self {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset
        for byte in name.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV-1a prime
        }
        Self::from_seed(hash)
    }

    /// Cosine similarity between two vectors.
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..DIM {
            let a = self.values[i];
            let b = other.values[i];
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        dot / denom
    }

    /// Element-wise addition (bundling).
    pub fn bundle(hvs: &[&HV]) -> Self {
        let mut result = Self::zero();
        for hv in hvs {
            for i in 0..DIM {
                result.values[i] += hv.values[i];
            }
        }
        // Normalize
        let norm: f32 = result.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut result.values {
                *v /= norm;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_similarity() {
        let v = HV::from_seed(42);
        assert!((v.similarity(&v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_seeds_orthogonal() {
        let a = HV::from_seed(1);
        let b = HV::from_seed(2);
        assert!(a.similarity(&b).abs() < 0.15);
    }

    #[test]
    fn test_deterministic() {
        let a = HV::from_name("Newton Second Law");
        let b = HV::from_name("Newton Second Law");
        assert!((a.similarity(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_name_encoding() {
        let newton = HV::from_name("Newton Second Law");
        let maxwell = HV::from_name("Maxwell Gauss Law");
        let sim = newton.similarity(&maxwell);
        assert!(
            sim.abs() < 0.3,
            "Different equations should be dissimilar: {}",
            sim
        );
    }
}
