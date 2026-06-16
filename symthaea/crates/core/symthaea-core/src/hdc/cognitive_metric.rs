// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Cognitive Metric — Non-Euclidean Geometric Similarity
//!
//! Implements a Riemannian metric tensor for the HDC vector space.
//! Uses 'Cognitive Mass' from useful concepts to warp the geometric
//! manifold, causing search and clustering to 'fall' toward established wisdom.
//!
//! This is a low-rank approximation of a 16,384D metric tensor to remain
//! computationally feasible.

use crate::hdc::unified_hv::ContinuousHV;

/// A point-mass in cognitive space that curves the manifold.
pub struct CognitiveMass {
    /// The concept vector (centroid or macro encoding)
    pub vector: ContinuousHV,
    /// The 'mass' or utility of this concept (higher = more curvature)
    pub mass: f64,
}

/// A Riemannian metric for warping HDC similarity.
pub struct CognitiveMetric {
    /// List of active cognitive masses curving the space
    pub masses: Vec<CognitiveMass>,
}

impl CognitiveMetric {
    pub fn new() -> Self {
        Self { masses: Vec::new() }
    }

    /// Add a concept as a gravitational mass.
    pub fn add_mass(&mut self, vector: ContinuousHV, mass: f64) {
        self.masses.push(CognitiveMass { vector, mass });
    }

    /// Compute warped dot product: A ·_g B = A·B + Σ mass_k * (A·M_k) * (B·M_k)
    pub fn warped_dot(&self, a: &ContinuousHV, b: &ContinuousHV) -> f64 {
        let base_dot = a.dot(b) as f64;

        let curvature: f64 = self
            .masses
            .iter()
            .map(|m| {
                let a_proj = a.dot(&m.vector) as f64;
                let b_proj = b.dot(&m.vector) as f64;
                m.mass * a_proj * b_proj
            })
            .sum();

        base_dot + curvature
    }

    /// Compute warped norm squared: ||A||_g^2 = A ·_g A
    pub fn warped_norm_sq(&self, a: &ContinuousHV) -> f64 {
        self.warped_dot(a, a)
    }

    /// Compute warped similarity (Riemannian Cosine Similarity)
    pub fn warped_similarity(&self, a: &ContinuousHV, b: &ContinuousHV) -> f64 {
        let dot = self.warped_dot(a, b);
        let norm_a = self.warped_norm_sq(a).sqrt();
        let norm_b = self.warped_norm_sq(b).sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

impl Default for CognitiveMetric {
    fn default() -> Self {
        Self::new()
    }
}
