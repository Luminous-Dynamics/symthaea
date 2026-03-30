// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::math::Vec3;
use std::f32::consts::PI;

/// Generate UV sphere mesh. Returns (interleaved vertices [pos.xyz, normal.xyz, uv.xy], indices).
pub fn generate_sphere(lat_segments: u32, lon_segments: u32, radius: f32) -> (Vec<f32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for lat in 0..=lat_segments {
        let theta = lat as f32 * PI / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=lon_segments {
            let phi = lon as f32 * 2.0 * PI / lon_segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = sin_theta * cos_phi;
            let y = cos_theta;
            let z = sin_theta * sin_phi;

            // Position
            vertices.push(x * radius);
            vertices.push(y * radius);
            vertices.push(z * radius);
            // Normal
            vertices.push(x);
            vertices.push(y);
            vertices.push(z);
            // UV
            vertices.push(lon as f32 / lon_segments as f32);
            vertices.push(lat as f32 / lat_segments as f32);
        }
    }

    let stride = lon_segments + 1;
    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let a = lat * stride + lon;
            let b = a + stride;

            indices.push(a);
            indices.push(b);
            indices.push(a + 1);

            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
        }
    }

    (vertices, indices)
}

/// Generate great-circle arc between two points on the sphere, elevated above surface.
/// Returns flat position buffer (xyz × segments+1) for GL_LINE_STRIP.
pub fn generate_arc(from: Vec3, to: Vec3, peak_height: f32, segments: u32) -> Vec<f32> {
    let mut positions = Vec::with_capacity((segments as usize + 1) * 3);

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        // Slerp along the great circle
        let point = from.slerp(to, t);
        // Radial elevation: parabolic arc peaking at midpoint
        let elevation = 1.0 + peak_height * 4.0 * t * (1.0 - t);
        let elevated = point.normalize() * elevation;

        positions.push(elevated.x);
        positions.push(elevated.y);
        positions.push(elevated.z);
    }

    positions
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
    let mut data = Vec::with_capacity((segments as usize + 1) * 4);

    // Seed from endpoint positions for deterministic noise
    let seed = from.x * 73.0 + from.y * 137.0 + to.z * 251.0;

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let point = from.slerp(to, t);
        let elevated_dir = point.normalize();
        let elevation = 1.0 + peak_height * 4.0 * t * (1.0 - t);

        // Organic mycelial noise: displace perpendicular to the arc path
        // Compute tangent and bitangent for displacement
        let tangent = if i < segments {
            let next = from.slerp(to, (i + 1) as f32 / segments as f32);
            (next - point).normalize()
        } else {
            let prev = from.slerp(to, (i - 1) as f32 / segments as f32);
            (point - prev).normalize()
        };
        let bitangent = elevated_dir.cross(tangent).normalize();

        // Deterministic pseudo-noise using nested sine waves
        let noise = (t * 7.0 + seed).sin() * 0.4
            + (t * 13.0 + seed * 2.3).sin() * 0.3
            + (t * 23.0 + seed * 0.7).sin() * 0.15;

        // Displacement: stronger in the middle, zero at endpoints
        let envelope = (t * PI).sin(); // parabolic envelope
        let displacement = bitangent * (noise * 0.012 * envelope);

        let final_pos = elevated_dir * elevation + displacement;

        data.push(final_pos.x);
        data.push(final_pos.y);
        data.push(final_pos.z);
        data.push(t);
    }

    data
}

/// Generate realistic starfield with spectral colors and Milky Way concentration.
/// Returns interleaved [pos.xyz, color.rgb, brightness] × count = 7 floats per star.
pub fn generate_starfield(count: u32, radius: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(count as usize * 7);

    let mut seed: u32 = 0xDEAD_BEEF;
    let next_f32 = |s: &mut u32| -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        (*s as f32) / u32::MAX as f32
    };

    for _ in 0..count {
        // Position: uniform on sphere with Milky Way concentration
        let u = next_f32(&mut seed) * 2.0 - 1.0;
        let theta = next_f32(&mut seed) * 2.0 * PI;

        // Milky Way band: concentrate 40% of stars near the galactic plane
        // The galactic plane is tilted ~60° from ecliptic
        let galactic_tilt: f32 = 1.1; // radians (~63°)
        let raw_y = u;
        // Bias toward galactic plane for some stars
        let milky_way_bias = next_f32(&mut seed);
        let y = if milky_way_bias < 0.4 {
            // Concentrate near galactic plane (y ≈ 0 after rotation)
            raw_y * 0.15  // compress toward equator
        } else {
            raw_y
        };

        let r = (1.0 - y * y).sqrt().max(0.01);
        let mut px = r * theta.cos() * radius;
        let mut py = y * radius;
        let mut pz = r * theta.sin() * radius;

        // Rotate for galactic tilt
        let (s_tilt, c_tilt) = galactic_tilt.sin_cos();
        let new_py = py * c_tilt - pz * s_tilt;
        let new_pz = py * s_tilt + pz * c_tilt;
        py = new_py;
        pz = new_pz;

        data.push(px);
        data.push(py);
        data.push(pz);

        // Spectral color based on stellar classification
        let spectral = next_f32(&mut seed);
        let (cr, cg, cb) = if spectral < 0.05 {
            // O/B type: blue-white (hot, rare)
            (0.7, 0.8, 1.0)
        } else if spectral < 0.15 {
            // A type: white
            (0.9, 0.92, 1.0)
        } else if spectral < 0.30 {
            // F type: yellow-white
            (1.0, 0.97, 0.85)
        } else if spectral < 0.55 {
            // G type: yellow (sun-like, most common)
            (1.0, 0.9, 0.7)
        } else if spectral < 0.80 {
            // K type: orange
            (1.0, 0.75, 0.5)
        } else {
            // M type: red (cool, common)
            (1.0, 0.6, 0.4)
        };
        data.push(cr);
        data.push(cg);
        data.push(cb);

        // Brightness: mostly dim, few bright (magnitude distribution)
        let mag = next_f32(&mut seed);
        let brightness = if mag < 0.02 {
            0.9 + next_f32(&mut seed) * 0.1  // very bright stars (2%)
        } else if mag < 0.10 {
            0.5 + next_f32(&mut seed) * 0.4  // medium bright (8%)
        } else {
            0.1 + next_f32(&mut seed) * 0.4  // dim stars (90%)
        };
        // Milky Way stars are slightly brighter as a cluster
        let mw_boost = if milky_way_bias < 0.4 { 1.3 } else { 1.0 };
        data.push(brightness * mw_boost);
    }

    data
}
