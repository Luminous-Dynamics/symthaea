// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial coordinate encoding via conjunctive basis binding.
//!
//! Grid positions are encoded as `x_basis.bind(&y_basis.permute(1))`,
//! producing unique conjunctive spatial representations.

use super::StimulusAdapter;
use symthaea_core::hdc::ContinuousHV;

/// A 2D grid coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPosition {
    pub x: u32,
    pub y: u32,
}

impl GridPosition {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    /// Manhattan distance to another position.
    pub fn distance(&self, other: &Self) -> u32 {
        self.x.abs_diff(other.x) + self.y.abs_diff(other.y)
    }
}

/// Encodes grid positions as conjunctive spatial bindings.
pub struct SpatialAdapter {
    /// Seed offset for X-axis basis vectors.
    pub x_seed_offset: u64,
    /// Seed offset for Y-axis basis vectors (different space from X).
    pub y_seed_offset: u64,
}

impl Default for SpatialAdapter {
    fn default() -> Self {
        Self {
            x_seed_offset: 10_000,
            y_seed_offset: 20_000,
        }
    }
}

impl SpatialAdapter {
    /// Create a basis vector for an X coordinate.
    pub fn x_basis(&self, x: u32, dim: usize) -> ContinuousHV {
        ContinuousHV::random(dim, self.x_seed_offset + x as u64)
    }

    /// Create a basis vector for a Y coordinate.
    pub fn y_basis(&self, y: u32, dim: usize) -> ContinuousHV {
        ContinuousHV::random(dim, self.y_seed_offset + y as u64)
    }
}

impl StimulusAdapter for SpatialAdapter {
    type Stimulus = GridPosition;

    fn encode(&self, pos: &GridPosition, dim: usize) -> ContinuousHV {
        let x_hv = self.x_basis(pos.x, dim);
        let y_hv = self.y_basis(pos.y, dim);
        x_hv.bind(&y_hv.permute(1))
    }
}

/// A visual object with color, shape, and optional size features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisualObject {
    pub color: u32,
    pub shape: u32,
    pub size: u32,
}

impl VisualObject {
    pub fn new(color: u32, shape: u32, size: u32) -> Self {
        Self { color, shape, size }
    }
}

/// Encodes visual objects via feature-conjunction binding.
///
/// `color_hv.bind(&shape_hv).bind(&size_hv)` produces a unique
/// conjunctive representation of the full feature set.
pub struct VisualObjectAdapter {
    pub color_seed: u64,
    pub shape_seed: u64,
    pub size_seed: u64,
}

impl Default for VisualObjectAdapter {
    fn default() -> Self {
        Self {
            color_seed: 30_000,
            shape_seed: 40_000,
            size_seed: 50_000,
        }
    }
}

impl StimulusAdapter for VisualObjectAdapter {
    type Stimulus = VisualObject;

    fn encode(&self, obj: &VisualObject, dim: usize) -> ContinuousHV {
        let color_hv = ContinuousHV::random(dim, self.color_seed + obj.color as u64);
        let shape_hv = ContinuousHV::random(dim, self.shape_seed + obj.shape as u64);
        let size_hv = ContinuousHV::random(dim, self.size_seed + obj.size as u64);
        color_hv.bind(&shape_hv).bind(&size_hv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_position_same_encoding() {
        let adapter = SpatialAdapter::default();
        let pos = GridPosition::new(3, 5);
        let a = adapter.encode(&pos, 512);
        let b = adapter.encode(&pos, 512);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_positions_dissimilar() {
        let adapter = SpatialAdapter::default();
        let a = adapter.encode(&GridPosition::new(0, 0), 512);
        let b = adapter.encode(&GridPosition::new(5, 5), 512);
        assert!(a.similarity(&b).abs() < 0.2);
    }

    #[test]
    fn test_visual_object_conjunction() {
        let adapter = VisualObjectAdapter::default();
        let obj_a = VisualObject::new(1, 2, 3);
        let obj_b = VisualObject::new(1, 2, 4); // same color+shape, different size
        let a = adapter.encode(&obj_a, 512);
        let b = adapter.encode(&obj_b, 512);
        // Partial feature overlap should be dissimilar (binding is multiplicative)
        assert!(a.similarity(&b).abs() < 0.3);
    }

    #[test]
    fn test_visual_same_object_identical() {
        let adapter = VisualObjectAdapter::default();
        let obj = VisualObject::new(1, 2, 3);
        let a = adapter.encode(&obj, 512);
        let b = adapter.encode(&obj, 512);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-5);
    }
}
