// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Self-contained HDC primitives.

pub const HDC_DIM: usize = 16384;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContinuousHV {
    pub values: Vec<f32>,
}

impl ContinuousHV {
    pub fn zero(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }
    pub fn random(dim: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let values = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { values }
    }
    pub fn l2_normalize(&mut self) {
        let sum_sq: f32 = self.values.iter().map(|&x| x * x).sum();
        let norm = sum_sq.sqrt();
        if norm > 1e-8 {
            for v in self.values.iter_mut() {
                *v /= norm;
            }
        }
    }
    pub fn lerp_in_place(&mut self, other: &Self, s: f32, o: f32) {
        for (i, v) in self.values.iter_mut().enumerate() {
            *v = s * *v + o * other.values[i];
        }
    }
    pub fn similarity(&self, other: &Self) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// Sparse-HDC representation for memory-efficient visual fields.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseHV {
    pub active_indices: Vec<(u32, f32)>,
    pub dim: usize,
}

impl SparseHV {
    pub fn new(dim: usize) -> Self {
        Self {
            active_indices: Vec::new(),
            dim,
        }
    }

    pub fn add(&mut self, index: u32, value: f32) {
        if value.abs() > 1e-4 {
            self.active_indices.push((index, value));
        }
    }

    pub fn similarity_with_continuous(&self, continuous: &ContinuousHV) -> f32 {
        self.active_indices
            .iter()
            .map(|&(idx, val)| val * continuous.values[idx as usize])
            .sum()
    }
}
