// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rendering-backend-agnostic math on raw `[f32; N]` arrays.
//!
//! Consumers convert to/from their native types:
//! - Leptos WebGL: `Vec3 { x, y, z }` (custom)
//! - Bevy: `glam::Vec3`
//! - Symtropy: `nalgebra::Vector3<f32>`

// ─── Vec3 helpers ────────────────────────────────────────────────

pub fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn vec3_length(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

pub fn vec3_length_sq(v: [f32; 3]) -> f32 {
    vec3_dot(v, v)
}

pub fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len < 1e-10 {
        return [0.0; 3];
    }
    vec3_scale(v, 1.0 / len)
}

pub fn vec3_lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    vec3_add(vec3_scale(a, 1.0 - t), vec3_scale(b, t))
}

pub fn vec3_slerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let an = vec3_normalize(a);
    let bn = vec3_normalize(b);
    let dot = vec3_dot(an, bn).clamp(-1.0, 1.0);
    let theta = dot.acos();
    if theta.abs() < 1e-6 {
        return vec3_lerp(a, b, t);
    }
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    vec3_add(vec3_scale(a, wa), vec3_scale(b, wb))
}

pub fn vec3_negate(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

// ─── Vec4 helpers ────────────────────────────────────────────────

pub fn vec4_from_vec3(v: [f32; 3], w: f32) -> [f32; 4] {
    [v[0], v[1], v[2], w]
}

pub fn vec4_xyz(v: [f32; 4]) -> [f32; 3] {
    [v[0], v[1], v[2]]
}

// ─── Mat4 (column-major [[f32; 4]; 4], m[col][row]) ─────────────

pub const MAT4_IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn mat4_perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    let range_inv = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (near + far) * range_inv, -1.0],
        [0.0, 0.0, near * far * 2.0 * range_inv, 0.0],
    ]
}

pub fn mat4_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = vec3_normalize(vec3_sub(target, eye));
    let s = vec3_normalize(vec3_cross(f, up));
    let u = vec3_cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-vec3_dot(s, eye), -vec3_dot(u, eye), vec3_dot(f, eye), 1.0],
    ]
}

pub fn mat4_translate(v: [f32; 3]) -> [[f32; 4]; 4] {
    let mut m = MAT4_IDENTITY;
    m[3][0] = v[0];
    m[3][1] = v[1];
    m[3][2] = v[2];
    m
}

pub fn mat4_rotate_x(angle: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_rotate_y(angle: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_rotate_z(angle: f32) -> [[f32; 4]; 4] {
    let (s, c) = angle.sin_cos();
    [
        [c, s, 0.0, 0.0],
        [-s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

pub fn mat4_transform_vec4(m: [[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
    ]
}

pub fn mat4_inverse(m: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] = m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[2][1] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
        + m[3][1] * (m[1][2] * m[2][3] - m[1][3] * m[2][2]);

    inv[1][0] = -(m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[2][0] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
        + m[3][0] * (m[1][2] * m[2][3] - m[1][3] * m[2][2]));

    inv[2][0] = m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
        - m[2][0] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
        + m[3][0] * (m[1][1] * m[2][3] - m[1][3] * m[2][1]);

    inv[3][0] = -(m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
        - m[2][0] * (m[1][1] * m[3][2] - m[1][2] * m[3][1])
        + m[3][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]));

    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;

    inv[0][1] = -(m[0][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[2][1] * (m[0][2] * m[3][3] - m[0][3] * m[3][2])
        + m[3][1] * (m[0][2] * m[2][3] - m[0][3] * m[2][2]));
    inv[1][1] = m[0][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[2][0] * (m[0][2] * m[3][3] - m[0][3] * m[3][2])
        + m[3][0] * (m[0][2] * m[2][3] - m[0][3] * m[2][2]);
    inv[2][1] = -(m[0][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
        - m[2][0] * (m[0][1] * m[3][3] - m[0][3] * m[3][1])
        + m[3][0] * (m[0][1] * m[2][3] - m[0][3] * m[2][1]));
    inv[3][1] = m[0][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
        - m[2][0] * (m[0][1] * m[3][2] - m[0][2] * m[3][1])
        + m[3][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]);

    inv[0][2] = m[0][1] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
        - m[1][1] * (m[0][2] * m[3][3] - m[0][3] * m[3][2])
        + m[3][1] * (m[0][2] * m[1][3] - m[0][3] * m[1][2]);
    inv[1][2] = -(m[0][0] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
        - m[1][0] * (m[0][2] * m[3][3] - m[0][3] * m[3][2])
        + m[3][0] * (m[0][2] * m[1][3] - m[0][3] * m[1][2]));
    inv[2][2] = m[0][0] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
        - m[1][0] * (m[0][1] * m[3][3] - m[0][3] * m[3][1])
        + m[3][0] * (m[0][1] * m[1][3] - m[0][3] * m[1][1]);
    inv[3][2] = -(m[0][0] * (m[1][1] * m[3][2] - m[1][2] * m[3][1])
        - m[1][0] * (m[0][1] * m[3][2] - m[0][2] * m[3][1])
        + m[3][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]));

    inv[0][3] = -(m[0][1] * (m[1][2] * m[2][3] - m[1][3] * m[2][2])
        - m[1][1] * (m[0][2] * m[2][3] - m[0][3] * m[2][2])
        + m[2][1] * (m[0][2] * m[1][3] - m[0][3] * m[1][2]));
    inv[1][3] = m[0][0] * (m[1][2] * m[2][3] - m[1][3] * m[2][2])
        - m[1][0] * (m[0][2] * m[2][3] - m[0][3] * m[2][2])
        + m[2][0] * (m[0][2] * m[1][3] - m[0][3] * m[1][2]);
    inv[2][3] = -(m[0][0] * (m[1][1] * m[2][3] - m[1][3] * m[2][1])
        - m[1][0] * (m[0][1] * m[2][3] - m[0][3] * m[2][1])
        + m[2][0] * (m[0][1] * m[1][3] - m[0][3] * m[1][1]));
    inv[3][3] = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
        + m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]);

    for c in 0..4 {
        for r in 0..4 {
            inv[c][r] *= inv_det;
        }
    }
    Some(inv)
}

/// Extract upper-left 3x3 as flat [f32; 9] in column-major order (for mat3 uniforms).
pub fn mat4_normal_matrix(m: [[f32; 4]; 4]) -> [f32; 9] {
    [
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2],
    ]
}

/// Flat [f32; 16] in column-major order for GPU upload.
pub fn mat4_as_f32_array(m: [[f32; 4]; 4]) -> [f32; 16] {
    [
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3],
    ]
}

// ─── Quaternion [x, y, z, w] ─────────────────────────────────────

pub const QUAT_IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

pub fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let (s, c) = half.sin_cos();
    let a = vec3_normalize(axis);
    [a[0] * s, a[1] * s, a[2] * s, c]
}

pub fn quat_rotate_vec3(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let u = [q[0], q[1], q[2]];
    let s = q[3];
    // q * v * q^-1 optimized
    vec3_add(
        vec3_add(
            vec3_scale(u, 2.0 * vec3_dot(u, v)),
            vec3_scale(v, s * s - vec3_dot(u, u)),
        ),
        vec3_scale(vec3_cross(u, v), 2.0 * s),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_vec3_normalize() {
        let v = vec3_normalize([3.0, 0.0, 4.0]);
        assert!((vec3_length(v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_cross() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = vec3_cross(x, y);
        assert!((z[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_slerp_endpoints() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let s0 = vec3_slerp(a, b, 0.0);
        let s1 = vec3_slerp(a, b, 1.0);
        assert!((s0[0] - 1.0).abs() < 1e-5);
        assert!((s1[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mat4_identity_mul() {
        let m = mat4_perspective(1.0, 1.5, 0.1, 100.0);
        let result = mat4_mul(MAT4_IDENTITY, m);
        for c in 0..4 {
            for r in 0..4 {
                assert!((result[c][r] - m[c][r]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_mat4_inverse_identity() {
        let inv = mat4_inverse(MAT4_IDENTITY).unwrap();
        for c in 0..4 {
            for r in 0..4 {
                assert!((inv[c][r] - MAT4_IDENTITY[c][r]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_mat4_inverse_roundtrip() {
        let m = mat4_look_at([3.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let inv = mat4_inverse(m).unwrap();
        let product = mat4_mul(m, inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i][j] - expected).abs() < 1e-4,
                    "mat4 inverse roundtrip failed at [{i}][{j}]: {} vs {expected}",
                    product[i][j]
                );
            }
        }
    }

    #[test]
    fn test_quat_rotate_x_axis_90_deg() {
        let q = quat_from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let v = quat_rotate_vec3(q, [1.0, 0.0, 0.0]);
        assert!((v[0]).abs() < 1e-5);
        assert!((v[1] - 1.0).abs() < 1e-5);
    }
}
