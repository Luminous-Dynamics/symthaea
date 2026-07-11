// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-section geometric properties (SI units, metres).

use std::f64::consts::PI;

/// Geometric properties of a beam/column cross-section.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Section {
    /// Cross-sectional area A (m²).
    pub area: f64,
    /// Second moment of area I about the bending axis (m⁴).
    pub moment_of_inertia: f64,
    /// Distance from neutral axis to extreme fibre c (m).
    pub extreme_fiber: f64,
}

impl Section {
    /// Solid rectangle, width `b`, height `h`, bending about the horizontal
    /// centroidal axis. `I = b·h³/12`, `c = h/2`.
    pub fn rectangular(b: f64, h: f64) -> Section {
        Section {
            area: b * h,
            moment_of_inertia: b * h.powi(3) / 12.0,
            extreme_fiber: h / 2.0,
        }
    }

    /// Solid circle of diameter `d`. `I = π·d⁴/64`, `c = d/2`.
    pub fn circular(d: f64) -> Section {
        Section {
            area: PI * d * d / 4.0,
            moment_of_inertia: PI * d.powi(4) / 64.0,
            extreme_fiber: d / 2.0,
        }
    }

    /// Hollow circular tube, outer/inner diameters `d_out`/`d_in`.
    pub fn hollow_circular(d_out: f64, d_in: f64) -> Section {
        Section {
            area: PI * (d_out * d_out - d_in * d_in) / 4.0,
            moment_of_inertia: PI * (d_out.powi(4) - d_in.powi(4)) / 64.0,
            extreme_fiber: d_out / 2.0,
        }
    }

    /// Elastic section modulus `S = I/c` (m³).
    pub fn section_modulus(&self) -> f64 {
        self.moment_of_inertia / self.extreme_fiber
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_inertia() {
        let s = Section::rectangular(0.05, 0.1);
        assert!((s.area - 0.005).abs() < 1e-12);
        assert!((s.moment_of_inertia - 4.166_666_67e-6).abs() < 1e-12);
        assert!((s.extreme_fiber - 0.05).abs() < 1e-12);
        assert!((s.section_modulus() - 8.333_333_3e-5).abs() < 1e-10);
    }

    #[test]
    fn circular_inertia() {
        let s = Section::circular(0.1);
        assert!((s.moment_of_inertia - PI * 0.1_f64.powi(4) / 64.0).abs() < 1e-15);
    }
}
