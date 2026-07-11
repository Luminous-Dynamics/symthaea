// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-structural
//!
//! A self-contained structural / mechanical engineering statics layer for
//! Symthaea: cross-section properties, Euler-Bernoulli beam analysis, axial
//! members, and Euler buckling.
//!
//! Fills a confirmed gap — the workspace had extensive robotics and materials
//! crates but **no structural/mechanical statics**: no beam bending, no section
//! properties, no buckling. All results here are closed-form and checked
//! against textbook hand calculations.
//!
//! ## Scope
//!
//! - Sections: rectangular, circular, hollow-circular; I, area, section modulus.
//! - Beams: the four canonical statically-determinate cases (cantilever ±
//!   end-point/UDL, simply-supported ± centre-point/UDL) → deflection, moment,
//!   bending stress, factor of safety.
//! - Members: axial stress/strain/elongation, Euler critical buckling load.
//! - Trusses: 2D pin-jointed solver by the method of joints (Gaussian
//!   elimination), with static-determinacy checks; member design check ties
//!   solved forces to axial stress and compression-member buckling safety.
//!
//! Not yet: 3D / frame solvers (stiffness matrix), indeterminate structures,
//! dynamics — the intended next direction.
//!
//! ## Example
//!
//! ```
//! use symthaea_structural::{Beam, LoadCase, Section, material::steel_a36};
//!
//! let beam = Beam {
//!     length: 2.0,
//!     section: Section::rectangular(0.05, 0.1),
//!     material: steel_a36(),
//! };
//! let r = beam.analyze(LoadCase::CantileverEndPoint(1000.0));
//! assert!((r.max_deflection - 0.0032).abs() < 1e-6); // 3.2 mm
//! assert!(r.factor_of_safety > 10.0);                // steel, well within yield
//! ```

pub mod beam;
pub mod design;
pub mod material;
pub mod member;
pub mod section;
pub mod truss;

pub use beam::{Beam, BeamResult, LoadCase};
pub use design::{DesignError, MemberCheck, MemberProperty, check_truss_members};
pub use material::Material;
pub use member::{axial_elongation, axial_strain, axial_stress, euler_buckling_load};
pub use section::Section;
pub use truss::{Load, Member, Node, Support, Truss, TrussError, TrussSolution};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::material::{aluminum_6061, steel_a36};

    #[test]
    fn worked_example_steel_vs_aluminum_cantilever() {
        // Same geometry and load, two materials. Aluminium (E≈1/3 of steel)
        // deflects ~2.9× more; stress is identical (geometry-only), so the
        // factor of safety differs only by yield strength.
        let section = Section::rectangular(0.05, 0.1);
        let load = LoadCase::CantileverEndPoint(1000.0);

        let steel = Beam {
            length: 2.0,
            section,
            material: steel_a36(),
        };
        let alu = Beam {
            length: 2.0,
            section,
            material: aluminum_6061(),
        };

        let rs = steel.analyze(load);
        let ra = alu.analyze(load);

        // Bending stress is material-independent (M·c/I).
        assert!((rs.max_bending_stress - ra.max_bending_stress).abs() < 1.0);

        // Aluminium deflects more, in the E ratio.
        let ratio = ra.max_deflection / rs.max_deflection;
        let e_ratio = steel_a36().youngs_modulus / aluminum_6061().youngs_modulus;
        assert!((ratio - e_ratio).abs() < 1e-6);

        // Both pass yield with margin.
        assert!(rs.factor_of_safety > 1.0);
        assert!(ra.factor_of_safety > 1.0);
    }

    #[test]
    fn slender_beam_fails_when_overloaded() {
        // A thin section under a heavy load should drop below FoS 1.0.
        let beam = Beam {
            length: 3.0,
            section: Section::rectangular(0.02, 0.02),
            material: steel_a36(),
        };
        let overload = beam.analyze(LoadCase::CantileverEndPoint(5000.0));
        assert!(
            overload.factor_of_safety < 1.0,
            "expected yield failure, FoS={}",
            overload.factor_of_safety
        );
    }
}
