// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Euler-Bernoulli beam analysis for standard support/load cases.
//!
//! Closed-form maximum deflection and bending moment for the four canonical
//! statically-determinate cases, plus derived bending stress and factor of
//! safety. All SI units (N, m, Pa).

use crate::material::Material;
use crate::section::Section;

/// A standard beam support + load configuration. Using explicit combined
/// variants makes invalid pairings (e.g. a "centre load" on a cantilever)
/// unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadCase {
    /// Cantilever, point load `P` (N) at the free end.
    CantileverEndPoint(f64),
    /// Cantilever, uniformly distributed load `w` (N/m) over the span.
    CantileverUdl(f64),
    /// Simply supported, point load `P` (N) at mid-span.
    SimplySupportedCenterPoint(f64),
    /// Simply supported, uniformly distributed load `w` (N/m).
    SimplySupportedUdl(f64),
}

/// A prismatic beam of a given length, section, and material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beam {
    /// Span length L (m).
    pub length: f64,
    pub section: Section,
    pub material: Material,
}

/// Analysis result for one load case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamResult {
    /// Maximum deflection (m).
    pub max_deflection: f64,
    /// Maximum bending moment (N·m).
    pub max_moment: f64,
    /// Maximum bending stress σ = M·c/I (Pa).
    pub max_bending_stress: f64,
    /// Factor of safety = yield / max bending stress.
    pub factor_of_safety: f64,
}

impl Beam {
    /// Maximum deflection for a load case (magnitude, m).
    pub fn max_deflection(&self, load: LoadCase) -> f64 {
        let l = self.length;
        let ei = self.material.youngs_modulus * self.section.moment_of_inertia;
        match load {
            LoadCase::CantileverEndPoint(p) => p * l.powi(3) / (3.0 * ei),
            LoadCase::CantileverUdl(w) => w * l.powi(4) / (8.0 * ei),
            LoadCase::SimplySupportedCenterPoint(p) => p * l.powi(3) / (48.0 * ei),
            LoadCase::SimplySupportedUdl(w) => 5.0 * w * l.powi(4) / (384.0 * ei),
        }
    }

    /// Maximum bending moment for a load case (magnitude, N·m).
    pub fn max_moment(&self, load: LoadCase) -> f64 {
        let l = self.length;
        match load {
            LoadCase::CantileverEndPoint(p) => p * l,
            LoadCase::CantileverUdl(w) => w * l * l / 2.0,
            LoadCase::SimplySupportedCenterPoint(p) => p * l / 4.0,
            LoadCase::SimplySupportedUdl(w) => w * l * l / 8.0,
        }
    }

    /// Maximum bending stress σ = M·c/I (Pa).
    pub fn max_bending_stress(&self, load: LoadCase) -> f64 {
        self.max_moment(load) / self.section.section_modulus()
    }

    /// Factor of safety against yielding in bending.
    pub fn factor_of_safety(&self, load: LoadCase) -> f64 {
        let stress = self.max_bending_stress(load);
        if stress <= 0.0 {
            f64::INFINITY
        } else {
            self.material.yield_strength / stress
        }
    }

    /// Full analysis for a load case.
    pub fn analyze(&self, load: LoadCase) -> BeamResult {
        BeamResult {
            max_deflection: self.max_deflection(load),
            max_moment: self.max_moment(load),
            max_bending_stress: self.max_bending_stress(load),
            factor_of_safety: self.factor_of_safety(load),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::steel_a36;

    fn steel_cantilever() -> Beam {
        Beam {
            length: 2.0,
            section: Section::rectangular(0.05, 0.1),
            material: steel_a36(),
        }
    }

    #[test]
    fn cantilever_end_point_hand_calc() {
        // L=2, P=1000, rect 0.05x0.1, E=200 GPa.
        // δ = PL³/3EI = 3.2 mm; M = PL = 2000 N·m; σ = M/S = 24 MPa; FoS ≈ 10.42.
        let b = steel_cantilever();
        let r = b.analyze(LoadCase::CantileverEndPoint(1000.0));
        assert!(
            (r.max_deflection - 0.0032).abs() < 1e-6,
            "δ={}",
            r.max_deflection
        );
        assert!((r.max_moment - 2000.0).abs() < 1e-6);
        assert!(
            (r.max_bending_stress - 24.0e6).abs() < 1e3,
            "σ={}",
            r.max_bending_stress
        );
        assert!((r.factor_of_safety - 10.4167).abs() < 1e-3);
    }

    #[test]
    fn simply_supported_center_is_stiffer_than_cantilever() {
        // Same P and geometry: SS-centre deflects far less (PL³/48EI vs PL³/3EI).
        let b = steel_cantilever();
        let cant = b.max_deflection(LoadCase::CantileverEndPoint(1000.0));
        let ss = b.max_deflection(LoadCase::SimplySupportedCenterPoint(1000.0));
        assert!(ss < cant);
        assert!((cant / ss - 16.0).abs() < 1e-6); // 48/3 = 16
    }

    #[test]
    fn udl_moment_formulas() {
        let b = steel_cantilever();
        // Cantilever UDL: M = wL²/2.
        assert!((b.max_moment(LoadCase::CantileverUdl(500.0)) - 500.0 * 4.0 / 2.0).abs() < 1e-9);
        // SS UDL: M = wL²/8.
        assert!(
            (b.max_moment(LoadCase::SimplySupportedUdl(500.0)) - 500.0 * 4.0 / 8.0).abs() < 1e-9
        );
    }
}
