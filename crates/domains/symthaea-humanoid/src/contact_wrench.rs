// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local six-dimensional contact-wrench feasibility for explicit support patches.
//!
//! This module evaluates a wrench against a validated contact patch without
//! granting solver or actuator authority. It is deliberately expressed in the
//! contact-local frame: z is the patch normal, x/y span the tangent plane, and
//! the wrench moment is taken about the patch origin.

use serde::{Deserialize, Serialize};

use crate::contact_patch::ContactPatchGeometryV1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchLocalV1 {
    /// Moment about the patch origin, expressed in the local contact frame.
    pub torque_local_nm: [f64; 3],
    /// Force expressed in the local contact frame. `force_local_n[2]` is normal.
    pub force_local_n: [f64; 3],
}

impl ContactWrenchLocalV1 {
    pub fn is_finite(&self) -> bool {
        self.torque_local_nm.iter().all(|value| value.is_finite())
            && self.force_local_n.iter().all(|value| value.is_finite())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchToleranceV1 {
    /// Numerical tolerance for force inequalities.
    pub force_n: f64,
    /// Numerical tolerance for moment/COP inequalities.
    pub moment_nm: f64,
}

impl Default for ContactWrenchToleranceV1 {
    fn default() -> Self {
        Self {
            force_n: 1.0e-8,
            moment_nm: 1.0e-8,
        }
    }
}

impl ContactWrenchToleranceV1 {
    pub fn validate(&self) -> bool {
        self.force_n.is_finite()
            && self.force_n >= 0.0
            && self.moment_nm.is_finite()
            && self.moment_nm >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchFeasibilityV1 {
    /// `f_n`; negative values violate unilateral contact.
    pub normal_force_margin_n: f64,
    /// `mu * f_n - hypot(f_x, f_y)`; this is the true circular Coulomb cone check.
    pub friction_margin_n: f64,
    /// `r_t * f_n - |tau_z|` using the patch's torsional-friction radius.
    pub torsional_margin_nm: f64,
    /// One inward half-space margin per patch edge after eliminating division by `f_n`.
    /// Each value has units N*m and is non-negative when the implied COP lies inside.
    pub cop_edge_margins_nm: Vec<f64>,
    /// Implied COP when normal force is numerically non-zero.
    pub center_of_pressure_local_xy_m: Option<[f64; 2]>,
    pub feasible: bool,
}

impl ContactWrenchFeasibilityV1 {
    pub fn minimum_cop_margin_nm(&self) -> f64 {
        self.cop_edge_margins_nm
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactWrenchInputError {
    InvalidPatch,
    NonFiniteWrench,
    InvalidTolerance,
}

/// Evaluate a local six-dimensional contact wrench against a convex support patch.
///
/// Constraints:
///
/// - unilateral contact: `f_n >= 0`;
/// - Coulomb friction: `sqrt(f_x^2 + f_y^2) <= mu * f_n`;
/// - COP inside the declared convex patch;
/// - torsional friction: `|tau_z| <= r_t * f_n`.
///
/// COP inequalities are evaluated without dividing by `f_n`. For a CCW edge
/// `a -> b`, let `n_in` be its inward unit normal. With
/// `p_cop = [-tau_y/f_n, tau_x/f_n]`, the inside test
/// `n_in dot (p_cop - a) >= 0` is multiplied by non-negative `f_n` to obtain
/// the linear moment inequality
/// `n_in dot ([-tau_y, tau_x] - a * f_n) >= 0`.
///
/// The returned report is diagnostic evidence only. It does not itself make a
/// QP constraint authoritative and does not claim physical correctness of the
/// geometry source.
pub fn evaluate_contact_wrench(
    patch: &ContactPatchGeometryV1,
    wrench: ContactWrenchLocalV1,
    tolerance: ContactWrenchToleranceV1,
) -> Result<ContactWrenchFeasibilityV1, ContactWrenchInputError> {
    if !patch.validate() {
        return Err(ContactWrenchInputError::InvalidPatch);
    }
    if !wrench.is_finite() {
        return Err(ContactWrenchInputError::NonFiniteWrench);
    }
    if !tolerance.validate() {
        return Err(ContactWrenchInputError::InvalidTolerance);
    }

    let [tau_x, tau_y, tau_z] = wrench.torque_local_nm;
    let [force_x, force_y, force_n] = wrench.force_local_n;

    let normal_force_margin_n = force_n;
    let friction_margin_n =
        patch.friction_coefficient * force_n - force_x.hypot(force_y);
    let torsional_margin_nm =
        patch.torsional_friction_radius_m * force_n - tau_z.abs();

    let cop_moment_xy_nm = [-tau_y, tau_x];
    let mut cop_edge_margins_nm = Vec::with_capacity(patch.vertices_local_xy_m.len());
    for index in 0..patch.vertices_local_xy_m.len() {
        let a = patch.vertices_local_xy_m[index];
        let b = patch.vertices_local_xy_m[(index + 1) % patch.vertices_local_xy_m.len()];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let edge_length = edge[0].hypot(edge[1]);
        // patch.validate() rejects degenerate/collinear geometry, so this is nonzero.
        let inward_unit = [-edge[1] / edge_length, edge[0] / edge_length];
        let relative_moment = [
            cop_moment_xy_nm[0] - a[0] * force_n,
            cop_moment_xy_nm[1] - a[1] * force_n,
        ];
        cop_edge_margins_nm.push(
            inward_unit[0] * relative_moment[0] + inward_unit[1] * relative_moment[1],
        );
    }

    let center_of_pressure_local_xy_m = if force_n.abs() > tolerance.force_n {
        Some([-tau_y / force_n, tau_x / force_n])
    } else {
        None
    };

    let feasible = normal_force_margin_n >= -tolerance.force_n
        && friction_margin_n >= -tolerance.force_n
        && torsional_margin_nm >= -tolerance.moment_nm
        && cop_edge_margins_nm
            .iter()
            .all(|margin| *margin >= -tolerance.moment_nm);

    Ok(ContactWrenchFeasibilityV1 {
        normal_force_margin_n,
        friction_margin_n,
        torsional_margin_nm,
        cop_edge_margins_nm,
        center_of_pressure_local_xy_m,
        feasible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact_patch::ContactPatchGeometrySource;
    use crate::multi_contact::ContactSite;

    fn patch() -> ContactPatchGeometryV1 {
        ContactPatchGeometryV1 {
            site: ContactSite::RightFoot,
            vertices_local_xy_m: vec![
                [-0.10, -0.04],
                [0.10, -0.04],
                [0.10, 0.04],
                [-0.10, 0.04],
            ],
            friction_coefficient: 0.8,
            torsional_friction_radius_m: 0.03,
            source: ContactPatchGeometrySource::MorphologyDeclaration,
            geometry_id: "synthetic-right-foot".to_string(),
        }
    }

    fn centered(normal_force_n: f64) -> ContactWrenchLocalV1 {
        ContactWrenchLocalV1 {
            torque_local_nm: [0.0; 3],
            force_local_n: [0.0, 0.0, normal_force_n],
        }
    }

    #[test]
    fn centered_compressive_wrench_is_feasible() {
        let report = evaluate_contact_wrench(
            &patch(),
            centered(500.0),
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(report.feasible);
        assert_eq!(report.center_of_pressure_local_xy_m, Some([0.0, 0.0]));
        assert!(report.minimum_cop_margin_nm() > 0.0);
    }

    #[test]
    fn off_center_wrench_recovers_exact_inside_cop() {
        let force_n = 500.0;
        let desired_cop = [0.05, 0.02];
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [desired_cop[1] * force_n, -desired_cop[0] * force_n, 0.0],
            force_local_n: [0.0, 0.0, force_n],
        };
        let report = evaluate_contact_wrench(
            &patch(),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(report.feasible);
        let recovered = report.center_of_pressure_local_xy_m.unwrap();
        assert!((recovered[0] - desired_cop[0]).abs() < 1.0e-12);
        assert!((recovered[1] - desired_cop[1]).abs() < 1.0e-12);
    }

    #[test]
    fn cop_outside_patch_is_rejected() {
        let force_n = 500.0;
        let outside_x = 0.12;
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0, -outside_x * force_n, 0.0],
            force_local_n: [0.0, 0.0, force_n],
        };
        let report = evaluate_contact_wrench(
            &patch(),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.minimum_cop_margin_nm() < 0.0);
    }

    #[test]
    fn true_coulomb_cone_rejects_excess_tangential_force() {
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0; 3],
            force_local_n: [401.0, 0.0, 500.0],
        };
        let report = evaluate_contact_wrench(
            &patch(),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.friction_margin_n < 0.0);
    }

    #[test]
    fn torsional_radius_has_correct_force_times_length_units() {
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0, 0.0, 15.1],
            force_local_n: [0.0, 0.0, 500.0],
        };
        let report = evaluate_contact_wrench(
            &patch(),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!((report.torsional_margin_nm + 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn tensile_normal_force_is_rejected() {
        let report = evaluate_contact_wrench(
            &patch(),
            centered(-1.0),
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.normal_force_margin_n < 0.0);
    }

    #[test]
    fn zero_normal_force_cannot_hide_nonzero_tangential_wrench() {
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0, 0.0, 0.01],
            force_local_n: [0.01, 0.0, 0.0],
        };
        let report = evaluate_contact_wrench(
            &patch(),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.friction_margin_n < 0.0);
        assert!(report.torsional_margin_nm < 0.0);
    }

    #[test]
    fn nonfinite_wrench_fails_as_invalid_evidence_not_infeasible_physics() {
        let mut wrench = centered(500.0);
        wrench.force_local_n[0] = f64::NAN;
        assert_eq!(
            evaluate_contact_wrench(
                &patch(),
                wrench,
                ContactWrenchToleranceV1::default(),
            ),
            Err(ContactWrenchInputError::NonFiniteWrench)
        );
    }
}
