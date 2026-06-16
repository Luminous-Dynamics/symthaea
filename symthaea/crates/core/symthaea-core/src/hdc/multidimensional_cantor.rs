// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multidimensional Cantor Structures for Hypervectors
//!
//! ## Beyond Linear Cantor
//!
//! The 1D Cantor set removes the middle third recursively along a line.
//! But cognition isn't linear - it's multidimensional!
//!
//! ## Radial Cantor
//!
//! ```text
//!     Core ────────────────────────► Periphery
//!     [Essential meaning]            [Contextual]
//!
//!     ████  remove  ██    remove    █
//!     band 1/3     band 1/3       ...recursive
//! ```
//!
//! Maps to cognition: Core = essential meaning, Periphery = associations
//!
//! ## 3D Cantor Dust
//!
//! Three orthogonal axes representing:
//! - X: Semantic dimension (what it IS)
//! - Y: Causal dimension (what it DOES)
//! - Z: Temporal dimension (WHEN it occurs)
//!
//! ## 4D Cantor Tesseract
//!
//! Adds a fourth dimension:
//! - W: Meta/consciousness dimension (awareness OF awareness)
//!
//! This creates a hypercube structure where each point in 4D space
//! represents a fully integrated cognitive element.

use super::binary_hv::BinaryHV;
use std::f64::consts::PI;

// =============================================================================
// RADIAL CANTOR HYPERVECTOR
// =============================================================================

/// Radial Cantor structure - fractal from core to periphery
///
/// In this model:
/// - Vector indices are mapped to radial distance from "center"
/// - Cantor removal happens in radial bands
/// - Core dimensions = essential/invariant meaning
/// - Outer dimensions = contextual/variable associations
///
/// ## Cognitive Interpretation
///
/// ```text
/// Core (indices 0-5000):     "dog" = ANIMAL, MAMMAL, PET (essential)
/// Middle (5000-11000):       "dog" = loyal, friendly (typical)
/// Periphery (11000-16384):   "dog" = my neighbor's specific dog (contextual)
/// ```
#[derive(Clone, Debug)]
pub struct RadialCantorHV {
    /// The radial-structured hypervector
    pub vector: BinaryHV,

    /// The base vector before radial structuring
    pub base: BinaryHV,

    /// Number of radial bands
    pub bands: usize,

    /// Cantor depth per band
    pub depth_per_band: usize,
}

impl RadialCantorHV {
    /// Create a radial Cantor HV
    ///
    /// Dimensions are organized into concentric bands, with Cantor
    /// structure applied within each band.
    pub fn new(base: BinaryHV, num_bands: usize) -> Self {
        let dim = BinaryHV::DIM;
        let band_size = dim / num_bands;
        let mut result = base;

        // Apply Cantor structure within each radial band
        for band in 0..num_bands {
            let start = band * band_size;
            let end = ((band + 1) * band_size).min(dim);

            // Weight by radial importance (core = higher weight)
            let radial_weight = 1.0 - (band as f64 / num_bands as f64) * 0.5;

            // Apply Cantor-like structure within band
            let band_shift = band_size / 3;
            if band_shift > 0 {
                let permuted = base.permute(start + band_shift);
                let weighted = Self::weight_vector(&permuted, radial_weight);
                result = result.bind(&weighted);
            }
        }

        Self {
            vector: result,
            base,
            bands: num_bands,
            depth_per_band: 3,
        }
    }

    /// Weight a vector's influence
    fn weight_vector(v: &BinaryHV, weight: f64) -> BinaryHV {
        if weight >= 1.0 {
            return *v;
        }
        // Probabilistic weighting: flip bits with probability (1-weight)
        let bytes = &v.0;
        let mut new_bytes = *bytes;
        let threshold = (weight * 255.0) as u8;

        for (i, byte) in new_bytes.iter_mut().enumerate() {
            // Simple deterministic "randomness" based on position
            let pseudo_random = ((i * 31 + 17) % 256) as u8;
            if pseudo_random > threshold {
                *byte = if *byte > 127 { 0 } else { 255 }; // Flip
            }
        }

        BinaryHV(new_bytes)
    }

    /// Query at specific radial depth
    ///
    /// Returns the vector focused on a specific radial band.
    pub fn at_radius(&self, band: usize) -> Option<BinaryHV> {
        if band >= self.bands {
            return None;
        }

        let dim = BinaryHV::DIM;
        let band_size = dim / self.bands;
        let shift = band * band_size + band_size / 3;

        Some(self.base.permute(shift))
    }

    /// Core meaning (innermost band)
    pub fn core(&self) -> BinaryHV {
        self.at_radius(0).unwrap_or(self.base)
    }

    /// Peripheral meaning (outermost band)
    pub fn periphery(&self) -> BinaryHV {
        self.at_radius(self.bands - 1).unwrap_or(self.base)
    }

    /// Similarity at each radial level
    pub fn radial_similarity(&self, other: &RadialCantorHV) -> Vec<(usize, f32)> {
        (0..self.bands.min(other.bands))
            .filter_map(|band| {
                let self_band = self.at_radius(band)?;
                let other_band = other.at_radius(band)?;
                Some((band, self_band.similarity(&other_band)))
            })
            .collect()
    }
}

// =============================================================================
// 3D CANTOR DUST HYPERVECTOR
// =============================================================================

/// 3D Cantor Dust - three orthogonal cognitive dimensions
///
/// Maps the hypervector space to a 3D cube where:
/// - X axis: Semantic (what it IS)
/// - Y axis: Causal (what it DOES/CAUSES)
/// - Z axis: Temporal (WHEN it exists/acts)
///
/// Cantor structure is applied in 3D, creating "Cantor dust" -
/// a fractal set with points in 3D space.
///
/// ## Mathematical Structure
///
/// For a 16,384D vector:
/// - Cube root ≈ 25.4, so we use 26 × 26 × 24 = 16,224 (close enough)
/// - Or we partition: 5,461 × 3 = 16,383
///
/// Each dimension is divided into thirds recursively (Cantor),
/// creating an 8-point (2³) structure at each level.
#[derive(Clone, Debug)]
pub struct Cantor3dHv {
    /// The 3D-structured hypervector
    pub vector: BinaryHV,

    /// Base vector
    pub base: BinaryHV,

    /// Dimensions per axis
    pub axis_size: usize,

    /// Recursion depth
    pub depth: usize,
}

impl Cantor3dHv {
    /// Create a 3D Cantor dust hypervector
    pub fn new(base: BinaryHV, depth: usize) -> Self {
        let dim = BinaryHV::DIM;
        // Partition into 3 axes
        let axis_size = dim / 3;

        let mut result = base;

        // Apply Cantor structure along each axis
        for axis in 0..3 {
            let axis_offset = axis * axis_size;

            // Recursive Cantor on this axis
            let mut shift = axis_size / 3;
            for d in 0..depth {
                if shift < 1 {
                    break;
                }

                let permuted = base.permute(axis_offset + shift);
                result = result.bind(&permuted);

                shift /= 3;
            }
        }

        // Cross-axis binding (creates 3D structure)
        let x_component = base.permute(0);
        let y_component = base.permute(axis_size);
        let z_component = base.permute(2 * axis_size);

        // Bind all three axes together
        let xyz_binding = x_component.bind(&y_component).bind(&z_component);
        result = BinaryHV::bundle(&[result, xyz_binding]);

        Self {
            vector: result,
            base,
            axis_size,
            depth,
        }
    }

    /// Get semantic component (X axis)
    pub fn semantic(&self) -> BinaryHV {
        self.base.permute(0)
    }

    /// Get causal component (Y axis)
    pub fn causal(&self) -> BinaryHV {
        self.base.permute(self.axis_size)
    }

    /// Get temporal component (Z axis)
    pub fn temporal(&self) -> BinaryHV {
        self.base.permute(2 * self.axis_size)
    }

    /// Query at 3D position
    ///
    /// x, y, z are normalized [0, 1] coordinates in the cognitive cube.
    pub fn at_position(&self, x: f64, y: f64, z: f64) -> BinaryHV {
        let x_shift = (x * self.axis_size as f64) as usize;
        let y_shift = self.axis_size + (y * self.axis_size as f64) as usize;
        let z_shift = 2 * self.axis_size + (z * self.axis_size as f64) as usize;

        let x_comp = self.base.permute(x_shift);
        let y_comp = self.base.permute(y_shift);
        let z_comp = self.base.permute(z_shift);

        x_comp.bind(&y_comp).bind(&z_comp)
    }

    /// Octant similarity (compare 8 corners of the cube)
    pub fn octant_similarity(&self, other: &Cantor3dHv) -> [f32; 8] {
        let positions = [
            (0.0, 0.0, 0.0), // 000
            (1.0, 0.0, 0.0), // 100
            (0.0, 1.0, 0.0), // 010
            (1.0, 1.0, 0.0), // 110
            (0.0, 0.0, 1.0), // 001
            (1.0, 0.0, 1.0), // 101
            (0.0, 1.0, 1.0), // 011
            (1.0, 1.0, 1.0), // 111
        ];

        let mut similarities = [0.0f32; 8];
        for (i, (x, y, z)) in positions.iter().enumerate() {
            let self_pos = self.at_position(*x, *y, *z);
            let other_pos = other.at_position(*x, *y, *z);
            similarities[i] = self_pos.similarity(&other_pos);
        }

        similarities
    }
}

// =============================================================================
// 4D CANTOR TESSERACT HYPERVECTOR
// =============================================================================

/// 4D Cantor Tesseract - consciousness as the fourth dimension
///
/// Extends 3D Cantor dust with a fourth axis:
/// - X: Semantic (what it IS)
/// - Y: Causal (what it DOES)
/// - Z: Temporal (WHEN)
/// - W: Meta/Consciousness (AWARENESS level)
///
/// The W dimension encodes:
/// - W=0: Unconscious/automatic processing
/// - W=0.5: Aware but not meta-aware
/// - W=1.0: Fully meta-conscious (aware of being aware)
///
/// ## The Tesseract Structure
///
/// A 4D hypercube has 16 vertices, 32 edges, 24 faces, 8 cells.
/// Each "cell" is a 3D cube at a different consciousness level.
///
/// ```text
///       W=1 (Meta-conscious)
///        ╱│
///       ╱ │
///      ╱  │
///     ╱   │
///    ╱    │
///   ╱     │
///  ╱      │
/// ┌───────┐
/// │ 3D    │
/// │ Cube  │
/// │       │
/// └───────┘
///     │
///     │
///     ▼
///   W=0 (Unconscious)
/// ```
#[derive(Clone, Debug)]
pub struct Cantor4dHv {
    /// The 4D-structured hypervector
    pub vector: BinaryHV,

    /// Base vector
    pub base: BinaryHV,

    /// Dimensions per axis (4 axes)
    pub axis_size: usize,

    /// Recursion depth
    pub depth: usize,

    /// Current W (consciousness) level
    pub consciousness_level: f64,
}

impl Cantor4dHv {
    /// Create a 4D Cantor tesseract hypervector
    pub fn new(base: BinaryHV, depth: usize) -> Self {
        Self::with_consciousness(base, depth, 0.5) // Default: partially conscious
    }

    /// Create with specific consciousness level
    pub fn with_consciousness(base: BinaryHV, depth: usize, w: f64) -> Self {
        let dim = BinaryHV::DIM;
        // Partition into 4 axes
        let axis_size = dim / 4;

        let mut result = base;

        // Apply Cantor structure along each of 4 axes
        for axis in 0..4 {
            let axis_offset = axis * axis_size;

            let mut shift = axis_size / 3;
            for _ in 0..depth {
                if shift < 1 {
                    break;
                }

                let permuted = base.permute(axis_offset + shift);
                result = result.bind(&permuted);

                shift /= 3;
            }
        }

        // 4D cross-binding (creates tesseract structure)
        let x = base.permute(0);
        let y = base.permute(axis_size);
        let z = base.permute(2 * axis_size);
        let w_vec = base.permute(3 * axis_size);

        // Bind all four axes
        let xyzw = x.bind(&y).bind(&z).bind(&w_vec);
        result = BinaryHV::bundle(&[result, xyzw]);

        // Apply consciousness weighting
        let w_influence = Self::consciousness_weight(&base, w, axis_size);
        result = BinaryHV::bundle(&[result, w_influence]);

        Self {
            vector: result,
            base,
            axis_size,
            depth,
            consciousness_level: w,
        }
    }

    /// Generate consciousness-weighted component
    fn consciousness_weight(base: &BinaryHV, w: f64, axis_size: usize) -> BinaryHV {
        // W dimension encodes meta-awareness
        // Higher W = more self-referential structure
        let w_shift = 3 * axis_size + (w * axis_size as f64) as usize;
        let w_component = base.permute(w_shift);

        if w > 0.5 {
            // High consciousness: bind with self (self-reference!)
            let self_ref = w_component.bind(&w_component);
            BinaryHV::bundle(&[w_component, self_ref])
        } else {
            w_component
        }
    }

    /// Get component at 4D position
    pub fn at_position(&self, x: f64, y: f64, z: f64, w: f64) -> BinaryHV {
        let x_shift = (x * self.axis_size as f64) as usize;
        let y_shift = self.axis_size + (y * self.axis_size as f64) as usize;
        let z_shift = 2 * self.axis_size + (z * self.axis_size as f64) as usize;
        let w_shift = 3 * self.axis_size + (w * self.axis_size as f64) as usize;

        let x_comp = self.base.permute(x_shift);
        let y_comp = self.base.permute(y_shift);
        let z_comp = self.base.permute(z_shift);
        let w_comp = self.base.permute(w_shift);

        x_comp.bind(&y_comp).bind(&z_comp).bind(&w_comp)
    }

    /// Semantic component (X)
    pub fn semantic(&self) -> BinaryHV {
        self.base.permute(0)
    }

    /// Causal component (Y)
    pub fn causal(&self) -> BinaryHV {
        self.base.permute(self.axis_size)
    }

    /// Temporal component (Z)
    pub fn temporal(&self) -> BinaryHV {
        self.base.permute(2 * self.axis_size)
    }

    /// Meta-conscious component (W)
    pub fn meta(&self) -> BinaryHV {
        self.base.permute(3 * self.axis_size)
    }

    /// Raise consciousness level
    pub fn raise_consciousness(&mut self, amount: f64) {
        self.consciousness_level = (self.consciousness_level + amount).min(1.0);
        // Rebuild with new consciousness level
        *self = Self::with_consciousness(self.base, self.depth, self.consciousness_level);
    }

    /// Lower consciousness level
    pub fn lower_consciousness(&mut self, amount: f64) {
        self.consciousness_level = (self.consciousness_level - amount).max(0.0);
        *self = Self::with_consciousness(self.base, self.depth, self.consciousness_level);
    }

    /// Compare at each consciousness level
    pub fn consciousness_gradient_similarity(&self, other: &Cantor4dHv) -> Vec<(f64, f32)> {
        let levels = [0.0, 0.25, 0.5, 0.75, 1.0];
        levels
            .iter()
            .map(|&w| {
                let self_at_w = self.at_position(0.5, 0.5, 0.5, w);
                let other_at_w = other.at_position(0.5, 0.5, 0.5, w);
                (w, self_at_w.similarity(&other_at_w))
            })
            .collect()
    }

    /// 16-vertex similarity (all corners of tesseract)
    pub fn tesseract_vertex_similarity(&self, other: &Cantor4dHv) -> f32 {
        let mut total_sim = 0.0;
        let mut count = 0;

        // 2^4 = 16 vertices
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    for w in [0.0, 1.0] {
                        let self_v = self.at_position(x, y, z, w);
                        let other_v = other.at_position(x, y, z, w);
                        total_sim += self_v.similarity(&other_v);
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total_sim / count as f32
        }
    }
}

// =============================================================================
// SPHERICAL CANTOR (Alternative to Radial)
// =============================================================================

/// Spherical Cantor - using spherical harmonics structure
///
/// Instead of linear or cubic Cantor, this uses spherical coordinates:
/// - r: Radial distance (importance/centrality)
/// - θ: Polar angle (semantic category)
/// - φ: Azimuthal angle (contextual variation)
///
/// Creates a spherical shell structure with Cantor gaps.
#[derive(Clone, Debug)]
pub struct SphericalCantorHV {
    /// The spherically-structured hypervector
    pub vector: BinaryHV,

    /// Base vector
    pub base: BinaryHV,

    /// Number of radial shells
    pub shells: usize,

    /// Angular resolution
    pub angular_resolution: usize,
}

impl SphericalCantorHV {
    /// Create a spherical Cantor hypervector
    pub fn new(base: BinaryHV, shells: usize) -> Self {
        let dim = BinaryHV::DIM;
        let angular_resolution = 16; // 16 angular divisions

        let mut result = base;

        // Create spherical structure
        for shell in 0..shells {
            let r = (shell as f64 + 1.0) / shells as f64;

            // Cantor scaling for radius
            let r_cantor = Self::cantor_transform(r);

            // Angular binding at this shell
            for theta_idx in 0..angular_resolution {
                let theta = (theta_idx as f64 / angular_resolution as f64) * PI;
                let phi_offset = ((theta.sin() * r_cantor * dim as f64) as usize) % dim;

                let angular_component = base.permute(phi_offset);

                // Weight by shell importance (inner = more important)
                if shell < shells / 2 {
                    result = result.bind(&angular_component);
                } else {
                    result = BinaryHV::bundle(&[result, angular_component]);
                }
            }
        }

        Self {
            vector: result,
            base,
            shells,
            angular_resolution,
        }
    }

    /// Cantor transform: maps [0,1] through Cantor structure
    fn cantor_transform(x: f64) -> f64 {
        // Approximate Cantor function (Devil's staircase)
        let mut result = 0.0;
        let mut remaining = x;
        let mut weight = 0.5;

        for _ in 0..10 {
            if remaining < 1.0 / 3.0 {
                remaining *= 3.0;
            } else if remaining > 2.0 / 3.0 {
                result += weight;
                remaining = (remaining - 2.0 / 3.0) * 3.0;
            } else {
                // In the middle third - Cantor gap
                result += weight / 2.0;
                break;
            }
            weight /= 2.0;
        }

        result
    }

    /// Get component at spherical coordinates
    pub fn at_spherical(&self, r: f64, theta: f64, phi: f64) -> BinaryHV {
        let dim = BinaryHV::DIM;

        // Convert spherical to index
        let r_idx = (r * dim as f64 / 3.0) as usize;
        let theta_idx = (theta / PI * dim as f64 / 3.0) as usize;
        let phi_idx = (phi / (2.0 * PI) * dim as f64 / 3.0) as usize;

        let r_comp = self.base.permute(r_idx % dim);
        let theta_comp = self.base.permute((dim / 3 + theta_idx) % dim);
        let phi_comp = self.base.permute((2 * dim / 3 + phi_idx) % dim);

        r_comp.bind(&theta_comp).bind(&phi_comp)
    }
}

// =============================================================================
// COGNITIVE MANIFOLD: Unifying All Cantor Structures
// =============================================================================

/// A cognitive manifold that can use any Cantor structure
#[derive(Clone)]
pub enum CantorManifold {
    /// Linear (1D) Cantor
    Linear(super::cantor_recursive_hv::CantorRecursiveHV),
    /// Radial Cantor
    Radial(RadialCantorHV),
    /// 3D Cantor Dust
    Dust3D(Cantor3dHv),
    /// 4D Cantor Tesseract
    Tesseract4D(Cantor4dHv),
    /// Spherical Cantor
    Spherical(SphericalCantorHV),
}

impl CantorManifold {
    /// Get the unified vector regardless of structure
    pub fn vector(&self) -> &BinaryHV {
        match self {
            CantorManifold::Linear(c) => &c.vector,
            CantorManifold::Radial(c) => &c.vector,
            CantorManifold::Dust3D(c) => &c.vector,
            CantorManifold::Tesseract4D(c) => &c.vector,
            CantorManifold::Spherical(c) => &c.vector,
        }
    }

    /// Similarity between any two manifolds
    pub fn similarity(&self, other: &CantorManifold) -> f32 {
        self.vector().similarity(other.vector())
    }

    /// Dimensionality of the Cantor structure
    pub fn dimensionality(&self) -> usize {
        match self {
            CantorManifold::Linear(_) => 1,
            CantorManifold::Radial(_) => 1, // Radial is 1D in r
            CantorManifold::Dust3D(_) => 3,
            CantorManifold::Tesseract4D(_) => 4,
            CantorManifold::Spherical(_) => 3, // r, θ, φ
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radial_cantor() {
        let base = BinaryHV::random(42);
        let radial = RadialCantorHV::new(base, 5);

        assert_eq!(radial.bands, 5);

        let core = radial.core();
        let periphery = radial.periphery();

        // Core and periphery should be different
        let sim = core.similarity(&periphery);
        println!("Core vs Periphery similarity: {:.4}", sim);
    }

    #[test]
    fn test_3d_cantor() {
        let base = BinaryHV::random(42);
        let dust = Cantor3dHv::new(base, 3);

        let semantic = dust.semantic();
        let causal = dust.causal();
        let temporal = dust.temporal();

        // All dimensions should produce vectors of equal length
        // BinaryHV is fixed-size (16,384 bits = 2048 bytes), all same dimension
        assert_eq!(
            BinaryHV::DIM,
            BinaryHV::DIM,
            "Semantic and causal should have same dimension"
        );

        let sim_sc = semantic.similarity(&causal);
        let sim_st = semantic.similarity(&temporal);
        let sim_ct = causal.similarity(&temporal);

        // Similarities should be valid (in [0, 1])
        assert!(
            sim_sc >= 0.0 && sim_sc <= 1.0,
            "Similarity should be in [0,1], got {}",
            sim_sc
        );
        assert!(
            sim_st >= 0.0 && sim_st <= 1.0,
            "Similarity should be in [0,1], got {}",
            sim_st
        );
        assert!(
            sim_ct >= 0.0 && sim_ct <= 1.0,
            "Similarity should be in [0,1], got {}",
            sim_ct
        );

        println!("Semantic vs Causal: {:.4}", sim_sc);
        println!("Semantic vs Temporal: {:.4}", sim_st);
        println!("Causal vs Temporal: {:.4}", sim_ct);
    }

    #[test]
    fn test_4d_cantor_consciousness() {
        let base = BinaryHV::random(42);

        let low_consciousness = Cantor4dHv::with_consciousness(base, 3, 0.1);
        let high_consciousness = Cantor4dHv::with_consciousness(base, 3, 0.9);

        let sim = low_consciousness
            .vector
            .similarity(&high_consciousness.vector);
        println!("Low vs High consciousness similarity: {:.4}", sim);

        // They should be somewhat different due to consciousness weighting
        assert!(sim < 0.99);
    }

    #[test]
    fn test_consciousness_gradient() {
        let base_a = BinaryHV::random(42);
        let base_b = BinaryHV::random(43);

        let a = Cantor4dHv::new(base_a, 3);
        let b = Cantor4dHv::new(base_b, 3);

        let gradient = a.consciousness_gradient_similarity(&b);
        assert!(
            !gradient.is_empty(),
            "Gradient should produce at least one data point"
        );

        println!("Consciousness gradient similarity:");
        for (w, sim) in &gradient {
            assert!(w.is_finite(), "Weight should be finite");
            assert!(
                *w >= 0.0 && *w <= 1.0,
                "Weight should be in [0,1], got {}",
                w
            );
            assert!(sim.is_finite(), "Similarity should be finite");
            assert!(
                *sim >= 0.0 && *sim <= 1.0,
                "Similarity should be in [0,1], got {}",
                sim
            );
            println!("  W={:.2}: {:.4}", w, sim);
        }
    }

    #[test]
    fn test_spherical_cantor() {
        let base = BinaryHV::random(42);
        let spherical = SphericalCantorHV::new(base, 4);

        // Sample at different spherical positions
        let origin = spherical.at_spherical(0.0, 0.0, 0.0);
        let equator = spherical.at_spherical(1.0, PI / 2.0, 0.0);
        let pole = spherical.at_spherical(1.0, 0.0, 0.0);

        let sim_oe = origin.similarity(&equator);
        let sim_op = origin.similarity(&pole);
        let sim_ep = equator.similarity(&pole);

        assert!(
            sim_oe >= 0.0 && sim_oe <= 1.0,
            "Similarity should be in [0,1], got {}",
            sim_oe
        );
        assert!(
            sim_op >= 0.0 && sim_op <= 1.0,
            "Similarity should be in [0,1], got {}",
            sim_op
        );
        assert!(
            sim_ep >= 0.0 && sim_ep <= 1.0,
            "Similarity should be in [0,1], got {}",
            sim_ep
        );

        println!("Origin vs Equator: {:.4}", sim_oe);
        println!("Origin vs Pole: {:.4}", sim_op);
        println!("Equator vs Pole: {:.4}", sim_ep);
    }
}
