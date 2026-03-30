// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::renderer::math::Vec3;
use std::f64::consts::PI;

/// Convert latitude/longitude (degrees) to 3D position on sphere.
/// Matches TerraGlobeWithSites.tsx lines 850-856 exactly.
pub fn lat_lon_to_xyz(lat: f64, lon: f64, radius: f64) -> Vec3 {
    let phi = (90.0 - lat) * (PI / 180.0);
    let theta = (lon + 180.0) * (PI / 180.0);
    Vec3::new(
        -(radius * phi.sin() * theta.cos()) as f32,
        (radius * phi.cos()) as f32,
        (radius * phi.sin() * theta.sin()) as f32,
    )
}

/// Marker size from capacity, matching TerraGlobe: log(capacity+1) * 0.0018, clamped [0.006, 0.022].
pub fn marker_size_from_capacity(capacity_mw: f64) -> f32 {
    let size = (capacity_mw + 1.0).ln() as f32 * 0.003;
    size.clamp(0.012, 0.035)
}

/// Peak height for arc elevation — low arcs hug the globe surface.
pub fn arc_peak_height(distance_km: f64) -> f32 {
    (0.015 + distance_km / 200000.0) as f32
}
