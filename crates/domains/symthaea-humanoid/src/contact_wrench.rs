// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local six-dimensional contact-wrench feasibility for explicit support patches.
//!
//! Geometry and friction are deliberately separate propositions. A patch defines
//! the support shape; `ContactInteractionLimitsV1` defines the limits of one
//! material/contact interaction at a stated evidence instant.

use serde::{Deserialize, Serialize};

use crate::contact_patch::ContactPatchGeometryV1;
use crate::multi_contact::ContactSite;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactInteractionLimitSource {
    /// Synthetic test/calibration fixture. Carries no simulator/hardware authority.
    SyntheticFixture,
    /// Limits derived from the exact simulator contact pair at the stated instant.
    SimulatorContactPair,
    /// Limits estimated from hardware sensing/system identification.
    HardwareEstimate,
    /// Explicitly conservative policy limit, narrower than available physical evidence.
    ConservativePolicy,
}

/// Friction/contact-interaction evidence for one contact site.
///
/// MuJoCo and real hardware may change these limits without changing the foot or
/// hand geometry. Keeping a separate identity prevents geometry provenance from
/// silently authorizing material/friction assumptions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactInteractionLimitsV1 {
    pub site: ContactSite,
    /// Dimensionless sliding/tangential friction coefficient.
    pub sliding_friction_coefficient: f64,
    /// Effective torsional-friction radius in metres, suitable for
    /// `|tau_z| <= r_t * f_n`.
    pub torsional_friction_radius_m: f64,
    pub source: ContactInteractionLimitSource,
    pub interaction_id: String,
    pub sampled_at_s: f64,
    pub confidence: f64,
}

impl ContactInteractionLimitsV1 {
    pub fn validate(&self) -> bool {
        self.sliding_friction_coefficient.is_finite()
            && self.sliding_friction_coefficient >= 0.0
            && self.torsional_friction_radius_m.is_finite()
            && self.torsional_friction_radius_m >= 0.0
            && !self.interaction_id.trim().is_empty()
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchToleranceV1 {
    pub force_n: f64,
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
    pub normal_force_margin_n: f64,
    /// `mu * f_n - hypot(f_x, f_y)`; circular Coulomb cone margin.
    pub friction_margin_n: f64,
    /// `r_t * f_n - |tau_z|`.
    pub torsional_margin_nm: f64,
    /// Inward half-space margin for every patch edge after eliminating COP division.
    pub cop_edge_margins_nm: Vec<f64>,
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
    InvalidInteractionLimits,
    SiteMismatch,
    NonFiniteWrench,
    InvalidTolerance,
}

/// Evaluate a local six-dimensional contact wrench against independent geometry
/// and interaction-limit evidence.
///
/// Constraints:
/// - unilateral contact: `f_n >= 0`;
/// - circular Coulomb friction: `sqrt(f_x^2 + f_y^2) <= mu * f_n`;
/// - COP inside the declared convex patch;
/// - torsional friction: `|tau_z| <= r_t * f_n`.
///
/// COP inequalities avoid division by `f_n`. For each CCW edge `a -> b` with
/// inward unit normal `n_in`, the usual COP condition is multiplied by the
/// non-negative normal force to obtain the linear wrench inequality
/// `n_in dot ([-tau_y, tau_x] - a * f_n) >= 0`.
pub fn evaluate_contact_wrench(
    patch: &ContactPatchGeometryV1,
    interaction: &ContactInteractionLimitsV1,
    wrench: ContactWrenchLocalV1,
    tolerance: ContactWrenchToleranceV1,
) -> Result<ContactWrenchFeasibilityV1, ContactWrenchInputError> {
    if !patch.validate() {
        return Err(ContactWrenchInputError::InvalidPatch);
    }
    if !interaction.validate() {
        return Err(ContactWrenchInputError::InvalidInteractionLimits);
    }
    if patch.site != interaction.site {
        return Err(ContactWrenchInputError::SiteMismatch);
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
    let friction_margin_n = interaction.sliding_friction_coefficient * force_n
        - force_x.hypot(force_y);
    let torsional_margin_nm = interaction.torsional_friction_radius_m * force_n - tau_z.abs();

    let cop_moment_xy_nm = [-tau_y, tau_x];
    let mut cop_edge_margins_nm = Vec::with_capacity(patch.vertices_local_xy_m.len());
    for index in 0..patch.vertices_local_xy_m.len() {
        let a = patch.vertices_local_xy_m[index];
        let b = patch.vertices_local_xy_m[(index + 1) % patch.vertices_local_xy_m.len()];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let edge_length = edge[0].hypot(edge[1]);
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

    fn patch(site: ContactSite) -> ContactPatchGeometryV1 {
        ContactPatchGeometryV1 {
            site,
            vertices_local_xy_m: vec![
                [-0.10, -0.04],
                [0.10, -0.04],
                [0.10, 0.04],
                [-0.10, 0.04],
            ],
            source: ContactPatchGeometrySource::MorphologyDeclaration,
            geometry_id: format!("synthetic-{site:?}"),
        }
    }

    fn limits(site: ContactSite) -> ContactInteractionLimitsV1 {
        ContactInteractionLimitsV1 {
            site,
            sliding_friction_coefficient: 0.8,
            torsional_friction_radius_m: 0.03,
            source: ContactInteractionLimitSource::SyntheticFixture,
            interaction_id: format!("synthetic-contact-{site:?}"),
            sampled_at_s: 1.0,
            confidence: 1.0,
        }
    }

    fn centered(normal_force_n: f64) -> ContactWrenchLocalV1 {
        ContactWrenchLocalV1 {
            torque_local_nm: [0.0; 3],
            force_local_n: [0.0, 0.0, normal_force_n],
        }
    }

    #[test]
    fn geometry_and_friction_are_independent_inputs() {
        let geometry = patch(ContactSite::RightFoot);
        let mut interaction = limits(ContactSite::RightFoot);
        assert!(evaluate_contact_wrench(
            &geometry,
            &interaction,
            centered(500.0),
            ContactWrenchToleranceV1::default(),
        )
        .unwrap()
        .feasible);
        interaction.sliding_friction_coefficient = 0.0;
        let slipping = ContactWrenchLocalV1 {
            torque_local_nm: [0.0; 3],
            force_local_n: [1.0, 0.0, 500.0],
        };
        assert!(!evaluate_contact_wrench(
            &geometry,
            &interaction,
            slipping,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap()
        .feasible);
        assert_eq!(geometry.geometry_id, "synthetic-RightFoot");
    }

    #[test]
    fn mismatched_site_interaction_is_rejected() {
        assert_eq!(
            evaluate_contact_wrench(
                &patch(ContactSite::RightFoot),
                &limits(ContactSite::LeftFoot),
                centered(500.0),
                ContactWrenchToleranceV1::default(),
            ),
            Err(ContactWrenchInputError::SiteMismatch)
        );
    }

    #[test]
    fn centered_compressive_wrench_is_feasible() {
        let report = evaluate_contact_wrench(
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
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
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
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
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0, -0.12 * force_n, 0.0],
            force_local_n: [0.0, 0.0, force_n],
        };
        let report = evaluate_contact_wrench(
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.minimum_cop_margin_nm() < 0.0);
    }

    #[test]
    fn circular_coulomb_cone_rejects_excess_tangential_force() {
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0; 3],
            force_local_n: [401.0, 0.0, 500.0],
        };
        let report = evaluate_contact_wrench(
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.friction_margin_n < 0.0);
    }

    #[test]
    fn torsional_limit_has_force_times_length_units() {
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0, 0.0, 15.1],
            force_local_n: [0.0, 0.0, 500.0],
        };
        let report = evaluate_contact_wrench(
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
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
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
            centered(-1.0),
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.normal_force_margin_n < 0.0);
    }

    #[test]
    fn zero_normal_force_cannot_hide_nonzero_tangential_or_torsional_wrench() {
        let wrench = ContactWrenchLocalV1 {
            torque_local_nm: [0.0, 0.0, 0.01],
            force_local_n: [0.01, 0.0, 0.0],
        };
        let report = evaluate_contact_wrench(
            &patch(ContactSite::RightFoot),
            &limits(ContactSite::RightFoot),
            wrench,
            ContactWrenchToleranceV1::default(),
        )
        .unwrap();
        assert!(!report.feasible);
        assert!(report.friction_margin_n < 0.0);
        assert!(report.torsional_margin_nm < 0.0);
    }

    #[test]
    fn nonfinite_wrench_is_invalid_evidence_not_physical_infeasibility() {
        let mut wrench = centered(500.0);
        wrench.force_local_n[0] = f64::NAN;
        assert_eq!(
            evaluate_contact_wrench(
                &patch(ContactSite::RightFoot),
                &limits(ContactSite::RightFoot),
                wrench,
                ContactWrenchToleranceV1::default(),
            ),
            Err(ContactWrenchInputError::NonFiniteWrench)
        );
    }
}
