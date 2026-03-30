// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Minimal linear algebra for WebGL — Vec3, Vec4, Mat4, Quat.
/// No external dependencies to keep WASM binary small.
use std::ops;

// ─── Vec3 ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        self * (1.0 / len)
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    pub fn slerp(self, other: Self, t: f32) -> Self {
        let dot = self.normalize().dot(other.normalize()).clamp(-1.0, 1.0);
        let theta = dot.acos();
        if theta.abs() < 1e-6 {
            return self.lerp(other, t);
        }
        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        self * a + other * b
    }

    pub fn negate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        self.negate()
    }
}

// ─── Vec4 ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_vec3(v: Vec3, w: f32) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w,
        }
    }

    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

// ─── Mat4 (column-major, matching WebGL) ─────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Mat4 {
    /// Column-major: m[col][row]
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let range_inv = 1.0 / (near - far);
        Self {
            m: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (near + far) * range_inv, -1.0],
                [0.0, 0.0, near * far * 2.0 * range_inv, 0.0],
            ],
        }
    }

    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            m: [
                [s.x, u.x, -f.x, 0.0],
                [s.y, u.y, -f.y, 0.0],
                [s.z, u.z, -f.z, 0.0],
                [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
            ],
        }
    }

    pub fn translate(v: Vec3) -> Self {
        let mut m = Self::IDENTITY;
        m.m[3][0] = v.x;
        m.m[3][1] = v.y;
        m.m[3][2] = v.z;
        m
    }

    pub fn rotate_x(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, s, 0.0],
                [0.0, -s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_y(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [
                [c, 0.0, -s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotate_z(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn mul(self, rhs: Self) -> Self {
        let mut out = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                out[c][r] = self.m[0][r] * rhs.m[c][0]
                    + self.m[1][r] * rhs.m[c][1]
                    + self.m[2][r] * rhs.m[c][2]
                    + self.m[3][r] * rhs.m[c][3];
            }
        }
        Self { m: out }
    }

    pub fn transform_vec4(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m[0][0] * v.x + self.m[1][0] * v.y + self.m[2][0] * v.z + self.m[3][0] * v.w,
            self.m[0][1] * v.x + self.m[1][1] * v.y + self.m[2][1] * v.z + self.m[3][1] * v.w,
            self.m[0][2] * v.x + self.m[1][2] * v.y + self.m[2][2] * v.z + self.m[3][2] * v.w,
            self.m[0][3] * v.x + self.m[1][3] * v.y + self.m[2][3] * v.z + self.m[3][3] * v.w,
        )
    }

    pub fn inverse(self) -> Option<Self> {
        let m = &self.m;
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
        Some(Self { m: inv })
    }

    /// Extract upper-left 3×3 as a flat [f32; 9] in column-major order (for mat3 uniform).
    pub fn normal_matrix(&self) -> [f32; 9] {
        // Normal matrix = transpose of inverse of upper-left 3×3.
        // For view*model matrices with uniform scale, this simplifies to the upper-left 3×3.
        [
            self.m[0][0], self.m[0][1], self.m[0][2],
            self.m[1][0], self.m[1][1], self.m[1][2],
            self.m[2][0], self.m[2][1], self.m[2][2],
        ]
    }

    /// Flat [f32; 16] in column-major order for `uniform_matrix4fv`.
    pub fn as_f32_array(&self) -> [f32; 16] {
        [
            self.m[0][0], self.m[0][1], self.m[0][2], self.m[0][3],
            self.m[1][0], self.m[1][1], self.m[1][2], self.m[1][3],
            self.m[2][0], self.m[2][1], self.m[2][2], self.m[2][3],
            self.m[3][0], self.m[3][1], self.m[3][2], self.m[3][3],
        ]
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ─── Quat ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let (s, c) = half.sin_cos();
        let a = axis.normalize();
        Self {
            x: a.x * s,
            y: a.y * s,
            z: a.z * s,
            w: c,
        }
    }

    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        // q * v * q^-1 optimized
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        u * 2.0 * u.dot(v) + v * (s * s - u.dot(u)) + u.cross(v) * 2.0 * s
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ─── Conversions to/from terra-atlas-core [f32; N] ──────────────

impl From<[f32; 3]> for Vec3 {
    fn from(a: [f32; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(v: Vec3) -> Self {
        [v.x, v.y, v.z]
    }
}

impl From<[f32; 4]> for Vec4 {
    fn from(a: [f32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }
}

impl From<Vec4> for [f32; 4] {
    fn from(v: Vec4) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}
