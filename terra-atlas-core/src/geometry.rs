// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Renderer-agnostic mesh generation: sphere, arcs, starfield.
//! All geometry output as flat `Vec<f32>` buffers for direct GPU upload.

use crate::math::*;
use std::f32::consts::PI;

/// Generate UV sphere mesh.
/// Returns (interleaved vertices [pos.xyz, normal.xyz, uv.xy], indices).
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
/// Returns flat position buffer (xyz * (segments+1)) for line strip rendering.
pub fn generate_arc(from: [f32; 3], to: [f32; 3], peak_height: f32, segments: u32) -> Vec<f32> {
    let mut positions = Vec::with_capacity((segments as usize + 1) * 3);

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let point = vec3_slerp(from, to, t);
        let elevation = 1.0 + peak_height * 4.0 * t * (1.0 - t);
        let elevated = vec3_scale(vec3_normalize(point), elevation);
        positions.push(elevated[0]);
        positions.push(elevated[1]);
        positions.push(elevated[2]);
    }

    positions
}

/// Generate arc with progress attribute for animated shaders.
/// Returns interleaved [pos.xyz, progress] * (segments+1).
/// Includes organic mycelial noise displacement perpendicular to the arc path.
pub fn generate_arc_with_progress(
    from: [f32; 3],
    to: [f32; 3],
    peak_height: f32,
    segments: u32,
) -> Vec<f32> {
    let mut data = Vec::with_capacity((segments as usize + 1) * 4);
    let seed = from[0] * 73.0 + from[1] * 137.0 + to[2] * 251.0;

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let point = vec3_slerp(from, to, t);
        let elevated_dir = vec3_normalize(point);
        let elevation = 1.0 + peak_height * 4.0 * t * (1.0 - t);

        // Tangent for perpendicular displacement
        let tangent = if i < segments {
            let next = vec3_slerp(from, to, (i + 1) as f32 / segments as f32);
            vec3_normalize(vec3_sub(next, point))
        } else {
            let prev = vec3_slerp(from, to, (i - 1) as f32 / segments as f32);
            vec3_normalize(vec3_sub(point, prev))
        };
        let bitangent = vec3_normalize(vec3_cross(elevated_dir, tangent));

        // Deterministic pseudo-noise
        let noise = (t * 7.0 + seed).sin() * 0.4
            + (t * 13.0 + seed * 2.3).sin() * 0.3
            + (t * 23.0 + seed * 0.7).sin() * 0.15;

        let envelope = (t * PI).sin();
        let displacement = vec3_scale(bitangent, noise * 0.012 * envelope);
        let final_pos = vec3_add(vec3_scale(elevated_dir, elevation), displacement);

        data.push(final_pos[0]);
        data.push(final_pos[1]);
        data.push(final_pos[2]);
        data.push(t);
    }

    data
}

/// Generate realistic starfield with spectral colors and Milky Way concentration.
/// Returns interleaved [pos.xyz, color.rgb, brightness] * count = 7 floats per star.
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
        let u = next_f32(&mut seed) * 2.0 - 1.0;
        let theta = next_f32(&mut seed) * 2.0 * PI;

        let galactic_tilt: f32 = 1.1;
        let raw_y = u;
        let milky_way_bias = next_f32(&mut seed);
        let y = if milky_way_bias < 0.4 {
            raw_y * 0.15
        } else {
            raw_y
        };

        let r = (1.0 - y * y).sqrt().max(0.01);
        let px = r * theta.cos() * radius;
        let mut py = y * radius;
        let mut pz = r * theta.sin() * radius;

        let (s_tilt, c_tilt) = galactic_tilt.sin_cos();
        let new_py = py * c_tilt - pz * s_tilt;
        let new_pz = py * s_tilt + pz * c_tilt;
        py = new_py;
        pz = new_pz;

        data.push(px);
        data.push(py);
        data.push(pz);

        let spectral = next_f32(&mut seed);
        let (cr, cg, cb) = if spectral < 0.05 {
            (0.7, 0.8, 1.0)
        } else if spectral < 0.15 {
            (0.9, 0.92, 1.0)
        } else if spectral < 0.30 {
            (1.0, 0.97, 0.85)
        } else if spectral < 0.55 {
            (1.0, 0.9, 0.7)
        } else if spectral < 0.80 {
            (1.0, 0.75, 0.5)
        } else {
            (1.0, 0.6, 0.4)
        };
        data.push(cr);
        data.push(cg);
        data.push(cb);

        let mag = next_f32(&mut seed);
        let brightness = if mag < 0.02 {
            0.9 + next_f32(&mut seed) * 0.1
        } else if mag < 0.10 {
            0.5 + next_f32(&mut seed) * 0.4
        } else {
            0.1 + next_f32(&mut seed) * 0.4
        };
        let mw_boost = if milky_way_bias < 0.4 { 1.3 } else { 1.0 };
        data.push(brightness * mw_boost);
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_vertex_count() {
        let (verts, indices) = generate_sphere(4, 4, 1.0);
        // (4+1) * (4+1) = 25 vertices, 8 floats each
        assert_eq!(verts.len(), 25 * 8);
        // 4 * 4 * 6 = 96 indices
        assert_eq!(indices.len(), 96);
    }

    #[test]
    fn test_arc_endpoint_preservation() {
        let from = [1.0, 0.0, 0.0];
        let to = [0.0, 1.0, 0.0];
        let arc = generate_arc(from, to, 0.0, 10);
        // First point should be near `from` (on unit sphere)
        let start = [arc[0], arc[1], arc[2]];
        let start_len = vec3_length(start);
        assert!((start_len - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_arc_with_progress_length() {
        let from = [1.0, 0.0, 0.0];
        let to = [0.0, 0.0, 1.0];
        let data = generate_arc_with_progress(from, to, 0.02, 20);
        assert_eq!(data.len(), 21 * 4);
        // First progress should be 0, last should be 1
        assert!((data[3] - 0.0).abs() < 1e-5);
        assert!((data[data.len() - 1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_starfield_count() {
        let data = generate_starfield(100, 50.0);
        assert_eq!(data.len(), 100 * 7);
    }

    #[test]
    fn test_starfield_deterministic() {
        let a = generate_starfield(50, 50.0);
        let b = generate_starfield(50, 50.0);
        assert_eq!(a, b, "starfield should be deterministic (same seed)");
    }
}
