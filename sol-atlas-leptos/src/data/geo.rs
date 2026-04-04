// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Thin wrappers over terra-atlas-core geo functions,
//! converting [f32; 3] to the local Vec3 type for renderer compatibility.

use crate::renderer::math::Vec3;

/// Convert latitude/longitude (degrees) to 3D position on sphere.
pub fn lat_lon_to_xyz(lat: f64, lon: f64, radius: f64) -> Vec3 {
    let p = sol_atlas_core::geo::lat_lon_to_xyz(lat, lon, radius);
    Vec3::new(p[0], p[1], p[2])
}

/// Marker size from capacity, logarithmic scaling, clamped.
pub fn marker_size_from_capacity(capacity_mw: f64) -> f32 {
    sol_atlas_core::geo::marker_size_from_capacity(capacity_mw)
}

/// Peak height for arc elevation above the globe surface.
pub fn arc_peak_height(distance_km: f64) -> f32 {
    sol_atlas_core::geo::arc_peak_height(distance_km)
}
