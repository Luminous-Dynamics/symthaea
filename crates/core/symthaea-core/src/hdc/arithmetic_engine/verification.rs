// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Adaptive verification threshold for HDC arithmetic.
///
/// The previous hardcoded 0.3 threshold was below the random baseline (0.5),
/// meaning it verified everything. This struct computes a statistically sound
/// threshold based on the dimensionality of BinaryHV vectors.
///
/// For 16,384-dim BinaryHV: random similarity ~ N(0.5, 1/sqrt(16384))
/// σ ≈ 0.0078, so 3σ threshold ≈ 0.5234
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VerificationThreshold {
    /// Random baseline similarity (0.5 for binary vectors)
    pub random_baseline: f32,
    /// Standard deviation: 1/sqrt(dimension)
    pub sigma: f32,
    /// Confidence multiplier (number of sigmas above baseline)
    pub k: f32,
}

impl VerificationThreshold {
    /// Standard threshold for 16,384-dimensional BinaryHV.
    pub fn for_binary_hv() -> Self {
        Self {
            random_baseline: 0.5,
            sigma: 1.0 / (16_384.0f32).sqrt(), // ≈ 0.0078
            k: 3.0,
        }
    }

    /// The base verification threshold: baseline + k * sigma.
    pub fn threshold(&self) -> f32 {
        self.random_baseline + self.k * self.sigma
    }

    /// Adaptive threshold that decreases slightly with construction depth.
    /// Deeper Peano constructions accumulate more noise, so we relax slightly.
    /// Minimum is still above random baseline.
    pub fn adaptive_threshold(&self, depth: u32) -> f32 {
        let base = self.threshold();
        // Relax by 0.5σ per depth level, but never below baseline + 1σ
        let relaxation = 0.5 * self.sigma * depth as f32;
        let min_threshold = self.random_baseline + self.sigma;
        (base - relaxation).max(min_threshold)
    }
}

impl Default for VerificationThreshold {
    fn default() -> Self {
        Self::for_binary_hv()
    }
}
