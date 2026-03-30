// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Thin wrappers over terra-atlas-core geometry generation,
//! converting Vec3 ↔ [f32; 3] at the boundary.

use super::math::Vec3;

/// Generate UV sphere mesh. Returns (interleaved vertices [pos.xyz, normal.xyz, uv.xy], indices).
pub fn generate_sphere(lat_segments: u32, lon_segments: u32, radius: f32) -> (Vec<f32>, Vec<u32>) {
    terra_atlas_core::geometry::generate_sphere(lat_segments, lon_segments, radius)
}

/// Generate great-circle arc between two points on the sphere, elevated above surface.
/// Returns flat position buffer (xyz × segments+1) for GL_LINE_STRIP.
pub fn generate_arc(from: Vec3, to: Vec3, peak_height: f32, segments: u32) -> Vec<f32> {
    terra_atlas_core::geometry::generate_arc(from.into(), to.into(), peak_height, segments)
}

/// Generate arc with progress attribute for animated shaders.
/// Returns interleaved [pos.xyz, progress] × (segments+1).
/// Includes organic mycelial noise displacement perpendicular to the arc path.
pub fn generate_arc_with_progress(
    from: Vec3,
    to: Vec3,
    peak_height: f32,
    segments: u32,
) -> Vec<f32> {
    terra_atlas_core::geometry::generate_arc_with_progress(
        from.into(),
        to.into(),
        peak_height,
        segments,
    )
}

/// Generate realistic starfield with spectral colors and Milky Way concentration.
/// Returns interleaved [pos.xyz, color.rgb, brightness] × count = 7 floats per star.
pub fn generate_starfield(count: u32, radius: f32) -> Vec<f32> {
    terra_atlas_core::geometry::generate_starfield(count, radius)
}
