// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural materials with elastic and strength properties (SI units).

/// An isotropic structural material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Human-readable name.
    pub name: &'static str,
    /// Young's modulus E (Pa).
    pub youngs_modulus: f64,
    /// Yield strength (Pa) — onset of permanent deformation.
    pub yield_strength: f64,
    /// Mass density (kg/m³).
    pub density: f64,
}

/// ASTM A36 structural steel.
pub fn steel_a36() -> Material {
    Material {
        name: "ASTM A36 steel",
        youngs_modulus: 200.0e9,
        yield_strength: 250.0e6,
        density: 7850.0,
    }
}

/// 6061-T6 aluminium alloy.
pub fn aluminum_6061() -> Material {
    Material {
        name: "6061-T6 aluminium",
        youngs_modulus: 68.9e9,
        yield_strength: 276.0e6,
        density: 2700.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steel_properties() {
        let s = steel_a36();
        assert!((s.youngs_modulus - 200.0e9).abs() < 1.0);
        assert!((s.yield_strength - 250.0e6).abs() < 1.0);
    }
}
