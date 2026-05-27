// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holographic Compression — Sparse Semantic Kernels for 64K states.
//!
//! Condenses ultra-high-resolution dilated hypervectors into
//! sparse representations for long-term storage.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// A sparse representation of a high-dimensional thought.
/// Only the most significant components (the 'kernel') are preserved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticKernel {
    pub dimension: usize,
    /// Indices of the top-N magnitude components.
    pub indices: Vec<u32>,
    /// Values at those indices.
    pub values: Vec<f32>,
}

impl SemanticKernel {
    /// Compress a full ContinuousHV into a sparse SemanticKernel.
    /// `top_n` determines the compression ratio.
    pub fn compress(hv: &ContinuousHV, top_n: usize) -> Self {
        let n = top_n.min(hv.dim());
        let mut indexed: Vec<(usize, f32)> = hv
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by absolute magnitude to find the most significant components
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(n);

        let mut indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for (i, v) in indexed {
            indices.push(i as u32);
            values.push(v);
        }

        Self {
            dimension: hv.dim(),
            indices,
            values,
        }
    }

    /// Decompress a sparse SemanticKernel back into a ContinuousHV.
    /// Missing components are restored as zeros.
    pub fn decompress(&self) -> ContinuousHV {
        let mut full_values = vec![0.0f32; self.dimension];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if (idx as usize) < self.dimension {
                full_values[idx as usize] = val;
            }
        }
        ContinuousHV::from_vec(full_values)
    }

    /// Measure the 'information density' of the kernel.
    pub fn density(&self) -> f32 {
        self.indices.len() as f32 / self.dimension as f32
    }
}
