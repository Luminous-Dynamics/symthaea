// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # ProbeMatrix
//!
//! A trained linear projection matrix **W** of shape `(hdc_dim × embedding_dim)`
//! stored in row-major order as a flat `Vec<f32>`.
//!
//! The projection computes:
//!
//! ```text
//! raw[k] = Σ_{j} W[k, j] · e[j]   for k in 0..hdc_dim
//! I(t)[k] = sign(raw[k])           (bipolar: +1.0 / -1.0)
//! ```
//!
//! The resulting [`ContinuousHV`] has exactly `hdc_dim` components, each ±1.0.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;
use symthaea_hdc_ltc::ContinuousHV;

// ─────────────────────────────────────────────────────────────────────────────
// ProbeMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Linear probe matrix W of shape `(hdc_dim × embedding_dim)` stored
/// in row-major order.
///
/// Each row `W[k, :]` is the weight vector for output dimension `k`.
/// Projection: `I(t) = sign(W @ embedding)`.
///
/// # Memory layout
///
/// `weights[k * embedding_dim + j]` = W[k, j]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeMatrix {
    /// Flat weights in row-major order: `hdc_dim` rows × `embedding_dim` cols.
    weights: Vec<f32>,
    /// Number of input embedding dimensions.
    embedding_dim: usize,
    /// Number of output HDC dimensions (rows of W).
    hdc_dim: usize,
}

impl ProbeMatrix {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a zero-initialized probe matrix.
    ///
    /// Useful for testing: projecting any embedding through a zero matrix
    /// gives a raw vector of all 0.0, which `sign` maps to all +1.0
    /// (positive convention for the zero case).
    pub fn from_zeros(embedding_dim: usize, hdc_dim: usize) -> Self {
        assert!(embedding_dim > 0, "embedding_dim must be > 0");
        assert!(hdc_dim > 0, "hdc_dim must be > 0");
        Self {
            weights: vec![0.0f32; hdc_dim * embedding_dim],
            embedding_dim,
            hdc_dim,
        }
    }

    /// Construct from a pre-allocated flat `Vec<f32>` of weights.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != hdc_dim * embedding_dim`.
    pub fn from_flat_f32(data: Vec<f32>, embedding_dim: usize, hdc_dim: usize) -> Self {
        assert!(embedding_dim > 0, "embedding_dim must be > 0");
        assert!(hdc_dim > 0, "hdc_dim must be > 0");
        assert_eq!(
            data.len(),
            hdc_dim * embedding_dim,
            "data length {} does not match hdc_dim ({}) * embedding_dim ({})",
            data.len(),
            hdc_dim,
            embedding_dim,
        );
        Self {
            weights: data,
            embedding_dim,
            hdc_dim,
        }
    }

    /// Load a probe matrix from a raw binary file of little-endian f32 values.
    ///
    /// The file must contain exactly `hdc_dim * embedding_dim` f32 values in
    /// row-major order (hdc_dim rows × embedding_dim cols).
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the file cannot be read or its length does not
    /// match the expected dimensions.
    pub fn load_flat_f32_file(
        path: impl AsRef<Path>,
        embedding_dim: usize,
        hdc_dim: usize,
    ) -> io::Result<Self> {
        assert!(embedding_dim > 0, "embedding_dim must be > 0");
        assert!(hdc_dim > 0, "hdc_dim must be > 0");

        let bytes = std::fs::read(path.as_ref())?;
        let expected_bytes = hdc_dim * embedding_dim * 4;

        if bytes.len() != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File has {} bytes; expected {} (hdc_dim={}, embedding_dim={})",
                    bytes.len(),
                    expected_bytes,
                    hdc_dim,
                    embedding_dim,
                ),
            ));
        }

        // Safety: we verified the exact byte count; f32 has align 4 and we
        // copy through from_le_bytes so no alignment assumption is made.
        let weights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self {
            weights,
            embedding_dim,
            hdc_dim,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Return the number of input embedding dimensions.
    #[inline]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Return the number of output HDC dimensions.
    #[inline]
    pub fn hdc_dim(&self) -> usize {
        self.hdc_dim
    }

    /// Read-only access to the flat weight buffer.
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    // ── Projection ────────────────────────────────────────────────────────────

    /// Project an embedding vector through the probe matrix.
    ///
    /// Computes `sign(W @ embedding)` where:
    /// - `W` has shape `(hdc_dim, embedding_dim)`
    /// - `sign(x) = +1.0` if `x >= 0`, else `-1.0`
    ///
    /// # Panics
    ///
    /// Panics if `embedding.len() != self.embedding_dim`.
    ///
    /// # Complexity
    ///
    /// O(hdc_dim × embedding_dim) — for the typical 16,384 × 768 case this is
    /// ~12.6 M multiply-add operations, completing in roughly 1–2 ms on a
    /// modern CPU without external BLAS.
    pub fn project(&self, embedding: &[f32]) -> ContinuousHV {
        assert_eq!(
            embedding.len(),
            self.embedding_dim,
            "embedding dimension mismatch: got {}, expected {}",
            embedding.len(),
            self.embedding_dim,
        );

        let mut values = Vec::with_capacity(self.hdc_dim);

        for k in 0..self.hdc_dim {
            // Row k of W starts at offset k * embedding_dim.
            let row_start = k * self.embedding_dim;
            let row = &self.weights[row_start..row_start + self.embedding_dim];

            let mut dot = 0.0f32;
            for j in 0..self.embedding_dim {
                dot += row[j] * embedding[j];
            }

            // Bipolar sign: 0 maps to +1.0 (positive convention).
            values.push(if dot >= 0.0 { 1.0f32 } else { -1.0f32 });
        }

        ContinuousHV::from_values(values)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn from_zeros_dimensions() {
        let probe = ProbeMatrix::from_zeros(768, 16_384);
        assert_eq!(probe.embedding_dim(), 768);
        assert_eq!(probe.hdc_dim(), 16_384);
        assert_eq!(probe.weights().len(), 768 * 16_384);
        assert!(probe.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn from_flat_f32_correct_length() {
        let data = vec![1.0f32; 4 * 3]; // 4 HDC dims, 3 embedding dims
        let probe = ProbeMatrix::from_flat_f32(data, 3, 4);
        assert_eq!(probe.embedding_dim(), 3);
        assert_eq!(probe.hdc_dim(), 4);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn from_flat_f32_wrong_length_panics() {
        // 10 values but 4*3 = 12 expected
        ProbeMatrix::from_flat_f32(vec![0.0; 10], 3, 4);
    }

    // ── Projection: zero matrix ───────────────────────────────────────────────

    /// A zero weight matrix produces raw dot products of 0 for any embedding.
    /// Our sign convention maps 0 → +1.0, so every output should be +1.0.
    #[test]
    fn zero_probe_gives_all_positive() {
        let probe = ProbeMatrix::from_zeros(4, 8);
        let embedding = vec![0.5, -0.3, 1.0, -1.0];
        let hv = probe.project(&embedding);

        assert_eq!(hv.dim(), 8);
        for &v in &hv.values {
            assert_eq!(v, 1.0f32, "Expected +1.0 for zero matrix projection");
        }
    }

    // ── Projection: known small matrix ────────────────────────────────────────

    /// Manually verify bipolar projection on a tiny (2 × 3) probe.
    ///
    /// W = [[1, -1,  0],   →  row 0 dot [1, 1, 1] = 0  → +1.0
    ///      [-1, -1, -1]]  →  row 1 dot [1, 1, 1] = -3 → -1.0
    #[test]
    fn known_matrix_correct_bipolar() {
        #[rustfmt::skip]
        let weights = vec![
            1.0f32, -1.0,  0.0,  // row 0
           -1.0f32, -1.0, -1.0,  // row 1
        ];
        let probe = ProbeMatrix::from_flat_f32(weights, 3, 2);
        let embedding = vec![1.0f32, 1.0, 1.0];

        let hv = probe.project(&embedding);
        assert_eq!(hv.dim(), 2);
        // row 0: 1*1 + (-1)*1 + 0*1 = 0  → sign → +1.0
        assert_eq!(hv.values[0], 1.0f32);
        // row 1: -1*1 + (-1)*1 + (-1)*1 = -3 → sign → -1.0
        assert_eq!(hv.values[1], -1.0f32);
    }

    /// Verify that strictly-positive dot products map to +1.0.
    #[test]
    fn positive_dot_product_gives_plus_one() {
        // Single row W = [1], embedding = [2] → dot = 2 → +1.0
        let probe = ProbeMatrix::from_flat_f32(vec![1.0f32], 1, 1);
        let hv = probe.project(&[2.0f32]);
        assert_eq!(hv.values[0], 1.0f32);
    }

    /// Verify that strictly-negative dot products map to -1.0.
    #[test]
    fn negative_dot_product_gives_minus_one() {
        // Single row W = [-1], embedding = [5] → dot = -5 → -1.0
        let probe = ProbeMatrix::from_flat_f32(vec![-1.0f32], 1, 1);
        let hv = probe.project(&[5.0f32]);
        assert_eq!(hv.values[0], -1.0f32);
    }

    /// Output is always bipolar (only ±1.0 values).
    #[test]
    fn output_is_bipolar_for_random_weights() {
        use symthaea_hdc_ltc::ContinuousHV;
        // Use a known-random HV to fill the weight matrix
        let hdc_dim = 64;
        let emb_dim = 32;
        let hv = ContinuousHV::new_random(hdc_dim * emb_dim, 42);
        let probe = ProbeMatrix::from_flat_f32(hv.values, emb_dim, hdc_dim);

        let embedding: Vec<f32> = (0..emb_dim).map(|i| (i as f32) * 0.01 - 0.16).collect();
        let out = probe.project(&embedding);

        assert_eq!(out.dim(), hdc_dim);
        for &v in &out.values {
            assert!(v == 1.0f32 || v == -1.0f32, "Non-bipolar value: {}", v);
        }
    }

    // ── File round-trip ───────────────────────────────────────────────────────

    #[test]
    fn load_flat_f32_file_roundtrip() {
        use std::io::Write;

        let hdc_dim = 4usize;
        let emb_dim = 3usize;
        let weights: Vec<f32> = vec![
            1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.9, -0.9, 0.8, -0.8,
        ];
        assert_eq!(weights.len(), hdc_dim * emb_dim);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.bin");

        // Write raw little-endian f32 bytes
        let mut f = std::fs::File::create(&path).unwrap();
        for &w in &weights {
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        drop(f);

        let probe = ProbeMatrix::load_flat_f32_file(&path, emb_dim, hdc_dim).unwrap();
        assert_eq!(probe.embedding_dim(), emb_dim);
        assert_eq!(probe.hdc_dim(), hdc_dim);
        for (a, b) in probe.weights().iter().zip(weights.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn load_flat_f32_file_wrong_size_returns_error() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");

        // Write only 4 bytes (one f32), but expect 4*3 = 12 f32s
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        drop(f);

        let result = ProbeMatrix::load_flat_f32_file(&path, 3, 4);
        assert!(result.is_err(), "Should fail on size mismatch");
    }

    // ── Serde ─────────────────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let probe = ProbeMatrix::from_flat_f32(vec![1.0, -1.0, 0.5, -0.5], 2, 2);
        let json = serde_json::to_string(&probe).unwrap();
        let restored: ProbeMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.embedding_dim(), probe.embedding_dim());
        assert_eq!(restored.hdc_dim(), probe.hdc_dim());
        assert_eq!(restored.weights(), probe.weights());
    }
}
