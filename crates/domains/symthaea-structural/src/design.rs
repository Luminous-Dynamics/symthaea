// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Truss member design checks: axial stress and Euler buckling safety.
//!
//! Ties the truss solver ([`crate::truss`]) to the member/buckling routines
//! ([`crate::member`]) into a real design-verification step: given a solved
//! truss and per-member section/material properties, report the axial stress in
//! every member and, for members in compression, the Euler buckling factor of
//! safety.

use crate::member::euler_buckling_load;
use crate::truss::{Truss, TrussSolution};

/// Section/material properties for one member.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemberProperty {
    /// Cross-sectional area (m²).
    pub area: f64,
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Second moment of area for buckling (m⁴).
    pub moment_of_inertia: f64,
}

/// Design-check outcome for one member.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemberCheck {
    pub member: usize,
    /// Axial force (N); +tension, −compression.
    pub force: f64,
    /// Member length (m).
    pub length: f64,
    /// Axial stress σ = F/A (Pa), signed.
    pub axial_stress: f64,
    /// True if the member is in compression.
    pub in_compression: bool,
    /// Euler critical buckling load (N) — only for compression members.
    pub buckling_load: Option<f64>,
    /// Buckling factor of safety `P_cr / |F|` — only for compression members.
    pub buckling_fos: Option<f64>,
}

/// Errors from a truss design check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DesignError {
    /// `properties.len()` does not match the member count.
    PropertyCountMismatch { members: usize, properties: usize },
}

/// Check every member for axial stress and (in compression) buckling safety.
///
/// `k_factor` is the effective-length factor for buckling (1.0 pinned-pinned).
pub fn check_truss_members(
    truss: &Truss,
    solution: &TrussSolution,
    properties: &[MemberProperty],
    k_factor: f64,
) -> Result<Vec<MemberCheck>, DesignError> {
    if properties.len() != truss.members.len() {
        return Err(DesignError::PropertyCountMismatch {
            members: truss.members.len(),
            properties: properties.len(),
        });
    }

    let mut checks = Vec::with_capacity(truss.members.len());
    for (idx, mem) in truss.members.iter().enumerate() {
        let ni = truss.nodes[mem.i];
        let nj = truss.nodes[mem.j];
        let length = ((nj.x - ni.x).powi(2) + (nj.y - ni.y).powi(2)).sqrt();
        let force = solution.member_forces[idx];
        let prop = properties[idx];
        let axial_stress = force / prop.area;
        let in_compression = force < 0.0;

        let (buckling_load, buckling_fos) = if in_compression {
            let p_cr = euler_buckling_load(
                prop.youngs_modulus,
                prop.moment_of_inertia,
                length,
                k_factor,
            );
            (Some(p_cr), Some(p_cr / force.abs()))
        } else {
            (None, None)
        };

        checks.push(MemberCheck {
            member: idx,
            force,
            length,
            axial_stress,
            in_compression,
            buckling_load,
            buckling_fos,
        });
    }
    Ok(checks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::truss::{Load, Member, Node, Truss};

    fn triangle() -> Truss {
        Truss {
            nodes: vec![
                Node::pin(0.0, 0.0),
                Node::roller_vertical(4.0, 0.0),
                Node::free(2.0, 3.0),
            ],
            members: vec![
                Member { i: 0, j: 1 }, // AB (tension)
                Member { i: 0, j: 2 }, // AC (compression)
                Member { i: 1, j: 2 }, // BC (compression)
            ],
            loads: vec![Load {
                node: 2,
                fx: 0.0,
                fy: -10.0,
            }],
        }
    }

    fn uniform_props(n: usize) -> Vec<MemberProperty> {
        vec![
            MemberProperty {
                area: 1e-4,
                youngs_modulus: 200e9,
                moment_of_inertia: 1e-8,
            };
            n
        ]
    }

    #[test]
    fn tension_member_has_no_buckling() {
        let t = triangle();
        let sol = t.solve().unwrap();
        let checks = check_truss_members(&t, &sol, &uniform_props(3), 1.0).unwrap();
        let ab = &checks[0];
        assert!(!ab.in_compression);
        assert!(ab.buckling_fos.is_none());
        assert!(ab.axial_stress > 0.0); // tension
    }

    #[test]
    fn compression_member_buckling_fos_hand_calc() {
        // AC: force −6.009 N, length √13 ≈ 3.6056 m, A=1e-4, I=1e-8, E=200 GPa.
        // P_cr = π²EI/(kL)² = 1518.4 N ⇒ FoS = 1518.4/6.009 ≈ 252.7.
        let t = triangle();
        let sol = t.solve().unwrap();
        let checks = check_truss_members(&t, &sol, &uniform_props(3), 1.0).unwrap();
        let ac = &checks[1];
        assert!(ac.in_compression);
        assert!((ac.length - 13.0_f64.sqrt()).abs() < 1e-9);
        assert!(
            (ac.buckling_load.unwrap() - 1518.4).abs() < 1.0,
            "Pcr={:?}",
            ac.buckling_load
        );
        assert!(
            (ac.buckling_fos.unwrap() - 252.7).abs() < 0.5,
            "FoS={:?}",
            ac.buckling_fos
        );
        assert!(ac.axial_stress < 0.0); // compression
    }

    #[test]
    fn property_count_mismatch_errors() {
        let t = triangle();
        let sol = t.solve().unwrap();
        assert!(matches!(
            check_truss_members(&t, &sol, &uniform_props(2), 1.0),
            Err(DesignError::PropertyCountMismatch { .. })
        ));
    }
}
