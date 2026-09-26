// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed facade for evidence-bearing use of the structural reference statics.

use crate::beam::{Beam, LoadCase};
use crate::design::{DesignError, MemberCheck, MemberProperty, check_truss_members};
use crate::material::Material;
use crate::member::{axial_elongation, axial_strain, axial_stress, euler_buckling_load};
use crate::section::Section;
use crate::truss::{Truss, TrussError, TrussSolution};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedStructuralError {
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
    Negative { field: &'static str },
    InvalidHollowSection,
    EmptyTopology,
    InvalidNodeIndex { kind: &'static str, index: usize, nodes: usize },
    ZeroLengthMember { member: usize },
    PropertyCountMismatch { members: usize, properties: usize },
    SolutionForceCountMismatch { members: usize, forces: usize },
    Truss(TrussError),
    Design(DesignError),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckedBeamResult {
    pub max_deflection: f64,
    pub max_moment: f64,
    pub max_bending_stress: f64,
    /// `None` for zero stress; zero load is not an infinite-safety claim.
    pub factor_of_safety: Option<f64>,
}

fn finite(field: &'static str, value: f64) -> Result<(), CheckedStructuralError> {
    value.is_finite().then_some(()).ok_or(CheckedStructuralError::NonFinite { field })
}

fn positive(field: &'static str, value: f64) -> Result<(), CheckedStructuralError> {
    finite(field, value)?;
    (value > 0.0).then_some(()).ok_or(CheckedStructuralError::NonPositive { field })
}

fn non_negative(field: &'static str, value: f64) -> Result<(), CheckedStructuralError> {
    finite(field, value)?;
    (value >= 0.0).then_some(()).ok_or(CheckedStructuralError::Negative { field })
}

pub fn validate_section(section: &Section) -> Result<(), CheckedStructuralError> {
    positive("section_area", section.area)?;
    positive("moment_of_inertia", section.moment_of_inertia)?;
    positive("extreme_fiber", section.extreme_fiber)?;
    positive("section_modulus", section.section_modulus())
}

pub fn try_rectangular_section(b: f64, h: f64) -> Result<Section, CheckedStructuralError> {
    positive("width", b)?;
    positive("height", h)?;
    let section = Section::rectangular(b, h);
    validate_section(&section)?;
    Ok(section)
}

pub fn try_circular_section(d: f64) -> Result<Section, CheckedStructuralError> {
    positive("diameter", d)?;
    let section = Section::circular(d);
    validate_section(&section)?;
    Ok(section)
}

pub fn try_hollow_circular_section(d_out: f64, d_in: f64) -> Result<Section, CheckedStructuralError> {
    positive("outer_diameter", d_out)?;
    positive("inner_diameter", d_in)?;
    if d_in >= d_out { return Err(CheckedStructuralError::InvalidHollowSection); }
    let section = Section::hollow_circular(d_out, d_in);
    validate_section(&section)?;
    Ok(section)
}

pub fn validate_material(material: &Material) -> Result<(), CheckedStructuralError> {
    positive("youngs_modulus", material.youngs_modulus)?;
    positive("yield_strength", material.yield_strength)?;
    positive("density", material.density)
}

fn validate_load(load: LoadCase) -> Result<(), CheckedStructuralError> {
    let v = match load {
        LoadCase::CantileverEndPoint(v) | LoadCase::CantileverUdl(v)
        | LoadCase::SimplySupportedCenterPoint(v) | LoadCase::SimplySupportedUdl(v) => v,
    };
    non_negative("load_magnitude", v)
}

pub fn try_analyze_beam(beam: &Beam, load: LoadCase) -> Result<CheckedBeamResult, CheckedStructuralError> {
    positive("beam_length", beam.length)?;
    validate_section(&beam.section)?;
    validate_material(&beam.material)?;
    validate_load(load)?;
    let d = beam.max_deflection(load);
    let m = beam.max_moment(load);
    let s = beam.max_bending_stress(load);
    finite("max_deflection_result", d)?;
    finite("max_moment_result", m)?;
    finite("max_bending_stress_result", s)?;
    let fos = if s == 0.0 { None } else {
        let v = beam.material.yield_strength / s;
        finite("factor_of_safety_result", v)?;
        Some(v)
    };
    Ok(CheckedBeamResult { max_deflection: d, max_moment: m, max_bending_stress: s, factor_of_safety: fos })
}

pub fn try_axial_stress(force: f64, area: f64) -> Result<f64, CheckedStructuralError> {
    finite("force", force)?;
    positive("area", area)?;
    let v = axial_stress(force, area);
    finite("axial_stress_result", v)?;
    Ok(v)
}

pub fn try_axial_strain(stress: f64, e: f64) -> Result<f64, CheckedStructuralError> {
    finite("stress", stress)?;
    positive("youngs_modulus", e)?;
    let v = axial_strain(stress, e);
    finite("axial_strain_result", v)?;
    Ok(v)
}

pub fn try_axial_elongation(force: f64, length: f64, area: f64, e: f64) -> Result<f64, CheckedStructuralError> {
    finite("force", force)?;
    positive("length", length)?;
    positive("area", area)?;
    positive("youngs_modulus", e)?;
    let v = axial_elongation(force, length, area, e);
    finite("axial_elongation_result", v)?;
    Ok(v)
}

pub fn try_euler_buckling_load(e: f64, i: f64, length: f64, k: f64) -> Result<f64, CheckedStructuralError> {
    positive("youngs_modulus", e)?;
    positive("moment_of_inertia", i)?;
    positive("length", length)?;
    positive("k_factor", k)?;
    let v = euler_buckling_load(e, i, length, k);
    finite("euler_buckling_result", v)?;
    Ok(v)
}

pub fn validate_truss(truss: &Truss) -> Result<(), CheckedStructuralError> {
    let n = truss.nodes.len();
    if n == 0 { return Err(CheckedStructuralError::EmptyTopology); }
    for node in &truss.nodes { finite("node_x", node.x)?; finite("node_y", node.y)?; }
    for (idx, member) in truss.members.iter().enumerate() {
        if member.i >= n { return Err(CheckedStructuralError::InvalidNodeIndex { kind: "member_i", index: member.i, nodes: n }); }
        if member.j >= n { return Err(CheckedStructuralError::InvalidNodeIndex { kind: "member_j", index: member.j, nodes: n }); }
        let a = truss.nodes[member.i];
        let b = truss.nodes[member.j];
        let q = (b.x - a.x).powi(2) + (b.y - a.y).powi(2);
        finite("member_length_squared", q)?;
        if q < 1e-24 { return Err(CheckedStructuralError::ZeroLengthMember { member: idx }); }
    }
    for load in &truss.loads {
        if load.node >= n { return Err(CheckedStructuralError::InvalidNodeIndex { kind: "load", index: load.node, nodes: n }); }
        finite("load_fx", load.fx)?;
        finite("load_fy", load.fy)?;
    }
    Ok(())
}

pub fn try_solve_truss(truss: &Truss) -> Result<TrussSolution, CheckedStructuralError> {
    validate_truss(truss)?;
    let solution = truss.solve().map_err(CheckedStructuralError::Truss)?;
    for v in &solution.member_forces { finite("truss_member_force_result", *v)?; }
    for r in &solution.reactions { finite("truss_reaction_fx_result", r.fx)?; finite("truss_reaction_fy_result", r.fy)?; }
    Ok(solution)
}

pub fn try_check_truss_members(
    truss: &Truss,
    solution: &TrussSolution,
    properties: &[MemberProperty],
    k: f64,
) -> Result<Vec<MemberCheck>, CheckedStructuralError> {
    validate_truss(truss)?;
    positive("k_factor", k)?;
    let m = truss.members.len();
    if properties.len() != m { return Err(CheckedStructuralError::PropertyCountMismatch { members: m, properties: properties.len() }); }
    if solution.member_forces.len() != m { return Err(CheckedStructuralError::SolutionForceCountMismatch { members: m, forces: solution.member_forces.len() }); }
    for v in &solution.member_forces { finite("solution_member_force", *v)?; }
    for p in properties {
        positive("member_area", p.area)?;
        positive("member_youngs_modulus", p.youngs_modulus)?;
        positive("member_moment_of_inertia", p.moment_of_inertia)?;
    }
    let checks = check_truss_members(truss, solution, properties, k).map_err(CheckedStructuralError::Design)?;
    for c in &checks {
        finite("member_length_result", c.length)?;
        finite("member_axial_stress_result", c.axial_stress)?;
        if let Some(v) = c.buckling_load { finite("member_buckling_load_result", v)?; }
        if let Some(v) = c.buckling_fos { finite("member_buckling_fos_result", v)?; }
    }
    Ok(checks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::steel_a36;
    use crate::truss::{Load, Member, Node};

    fn triangle() -> Truss {
        Truss {
            nodes: vec![Node::pin(0.0, 0.0), Node::roller_vertical(4.0, 0.0), Node::free(2.0, 3.0)],
            members: vec![Member { i: 0, j: 1 }, Member { i: 0, j: 2 }, Member { i: 1, j: 2 }],
            loads: vec![Load { node: 2, fx: 0.0, fy: -10.0 }],
        }
    }

    #[test]
    fn invalid_geometry_and_negative_load_fail_closed() {
        assert!(try_rectangular_section(0.0, 0.1).is_err());
        assert!(try_hollow_circular_section(0.1, 0.1).is_err());
        let beam = Beam { length: 2.0, section: try_rectangular_section(0.05, 0.1).unwrap(), material: steel_a36() };
        assert!(matches!(try_analyze_beam(&beam, LoadCase::CantileverEndPoint(-1.0)), Err(CheckedStructuralError::Negative { .. })));
    }

    #[test]
    fn zero_load_has_no_infinite_safety_claim() {
        let beam = Beam { length: 2.0, section: try_rectangular_section(0.05, 0.1).unwrap(), material: steel_a36() };
        let r = try_analyze_beam(&beam, LoadCase::CantileverEndPoint(0.0)).unwrap();
        assert_eq!(r.factor_of_safety, None);
    }

    #[test]
    fn invalid_truss_index_rejects_before_legacy_solver() {
        let mut t = triangle();
        t.members[0].j = 99;
        assert!(matches!(try_solve_truss(&t), Err(CheckedStructuralError::InvalidNodeIndex { .. })));
    }

    #[test]
    fn mismatched_solution_rejects_before_member_check() {
        let t = triangle();
        let mut s = try_solve_truss(&t).unwrap();
        s.member_forces.pop();
        let p = vec![MemberProperty { area: 1e-4, youngs_modulus: 200e9, moment_of_inertia: 1e-8 }; 3];
        assert!(matches!(try_check_truss_members(&t, &s, &p, 1.0), Err(CheckedStructuralError::SolutionForceCountMismatch { .. })));
    }
}
