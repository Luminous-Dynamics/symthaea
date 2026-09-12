// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cabibbo-Marinari-style SU(3) Metropolis update primitives.
//!
//! This module builds on `lattice_gauge` and implements one deterministic
//! subgroup proposal/accept-reject step at a time. Random-number generation and
//! sweep scheduling are deliberately external so their probability law can be
//! qualified independently.
//!
//! Authority boundary: a valid local update is not evidence of equilibrium,
//! ergodicity, thermalization, sufficient effective sample size, or continuum
//! physics.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_identity, su3_mul, su3_trace,
    validate_su3,
};
use crate::symmetry_groups::Complex;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Su2Subgroup {
    Pair01,
    Pair02,
    Pair12,
}

impl Su2Subgroup {
    pub const ALL: [Self; 3] = [Self::Pair01, Self::Pair02, Self::Pair12];

    pub fn indices(self) -> (usize, usize) {
        match self {
            Self::Pair01 => (0, 1),
            Self::Pair02 => (0, 2),
            Self::Pair12 => (1, 2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su2SubgroupProposal {
    pub subgroup: Su2Subgroup,
    /// Axis in su(2); normalized internally.
    pub axis: [f64; 3],
    /// Rotation angle in radians. A symmetric proposal law must give the
    /// corresponding negative angle / inverse move equal proposal density.
    pub angle: f64,
}

impl Su2SubgroupProposal {
    pub fn inverse(self) -> Self {
        Self {
            angle: -self.angle,
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetropolisStepResult {
    pub accepted: bool,
    pub delta_action: f64,
    pub acceptance_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeMetropolisError {
    Gauge(LatticeGaugeError),
    InvalidProposalAxis([f64; 3]),
    InvalidProposalAngle(f64),
    InvalidUniformDraw(f64),
    NonFiniteActionDelta(f64),
}

impl From<LatticeGaugeError> for LatticeMetropolisError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

pub fn embedded_su2_rotation(
    proposal: Su2SubgroupProposal,
) -> Result<Su3Matrix, LatticeMetropolisError> {
    if !proposal.angle.is_finite() {
        return Err(LatticeMetropolisError::InvalidProposalAngle(proposal.angle));
    }
    let norm_sq = proposal.axis.iter().map(|x| x * x).sum::<f64>();
    if !norm_sq.is_finite() || norm_sq <= 0.0 {
        return Err(LatticeMetropolisError::InvalidProposalAxis(proposal.axis));
    }
    let norm = norm_sq.sqrt();
    let [nx, ny, nz] = proposal.axis.map(|x| x / norm);
    let c = proposal.angle.cos();
    let s = proposal.angle.sin();

    // exp(i theta n.sigma)
    let u00 = Complex::new(c, s * nz);
    let u01 = Complex::new(s * ny, s * nx);
    let u10 = Complex::new(-s * ny, s * nx);
    let u11 = Complex::new(c, -s * nz);

    let mut out = su3_identity();
    let (i, j) = proposal.subgroup.indices();
    out[i][i] = u00;
    out[i][j] = u01;
    out[j][i] = u10;
    out[j][j] = u11;
    validate_su3(&out, 1e-12)?;
    Ok(out)
}

/// Wilson-action contribution of the unique plaquettes touching one link.
///
/// The set is deduplicated so degenerate extents (for example length 1) do not
/// double-count a canonical plaquette.
pub fn affected_wilson_action(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    beta: f64,
) -> Result<f64, LatticeMetropolisError> {
    if mu >= 4 {
        return Err(LatticeGaugeError::InvalidDirection(mu).into());
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err(LatticeGaugeError::InvalidBeta(beta).into());
    }

    let mut plaquettes = BTreeSet::new();
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let (a, b) = if mu < nu { (mu, nu) } else { (nu, mu) };
        plaquettes.insert((site, a, b));
        plaquettes.insert((field.shift(site, nu, -1)?, a, b));
    }

    let mut action = 0.0;
    for (base, a, b) in plaquettes {
        let normalized_trace = su3_trace(&field.plaquette(base, a, b)?).re / 3.0;
        action += beta * (1.0 - normalized_trace);
    }
    Ok(action)
}

pub fn metropolis_acceptance_probability(
    delta_action: f64,
) -> Result<f64, LatticeMetropolisError> {
    if !delta_action.is_finite() {
        return Err(LatticeMetropolisError::NonFiniteActionDelta(delta_action));
    }
    Ok(if delta_action <= 0.0 {
        1.0
    } else {
        (-delta_action).exp()
    })
}

/// Apply one deterministic subgroup proposal and Metropolis decision.
///
/// `uniform_draw` must be supplied by an independently qualified RNG/proposal
/// scheduler. This function assumes the proposal law is symmetric between the
/// supplied proposal and its inverse; otherwise a Hastings ratio is required.
pub fn metropolis_subgroup_step(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    beta: f64,
    proposal: Su2SubgroupProposal,
    uniform_draw: f64,
) -> Result<MetropolisStepResult, LatticeMetropolisError> {
    if !uniform_draw.is_finite() || !(0.0..1.0).contains(&uniform_draw) {
        return Err(LatticeMetropolisError::InvalidUniformDraw(uniform_draw));
    }

    let before = affected_wilson_action(field, site, mu, beta)?;
    let original = *field.link(site, mu)?;
    let rotation = embedded_su2_rotation(proposal)?;
    let candidate = su3_mul(&rotation, &original);
    field.set_link(site, mu, candidate)?;

    let after = affected_wilson_action(field, site, mu, beta)?;
    let delta_action = after - before;
    let acceptance_probability = metropolis_acceptance_probability(delta_action)?;
    let accepted = uniform_draw < acceptance_probability;

    if !accepted {
        field.set_link(site, mu, original)?;
    }

    Ok(MetropolisStepResult {
        accepted,
        delta_action,
        acceptance_probability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_gauge::{su3_dagger, su3_diagonal, su3_unitarity_error};

    fn matrix_max_error(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
        let mut max_error: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let dr = a[i][j].re - b[i][j].re;
                let di = a[i][j].im - b[i][j].im;
                max_error = max_error.max((dr * dr + di * di).sqrt());
            }
        }
        max_error
    }

    #[test]
    fn all_subgroup_proposals_are_su3_and_reversible() {
        for subgroup in Su2Subgroup::ALL {
            let proposal = Su2SubgroupProposal {
                subgroup,
                axis: [1.0, 2.0, 3.0],
                angle: 0.2,
            };
            let forward = embedded_su2_rotation(proposal).unwrap();
            let inverse = embedded_su2_rotation(proposal.inverse()).unwrap();
            assert!(su3_unitarity_error(&forward) < 1e-12);
            assert!(matrix_max_error(&inverse, &su3_dagger(&forward)) < 1e-12);
        }
    }

    #[test]
    fn accepted_step_matches_lqcd_011_oracle() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        let proposal = Su2SubgroupProposal {
            subgroup: Su2Subgroup::Pair01,
            axis: [1.0, 2.0, 3.0],
            angle: 0.2,
        };
        let before = field.wilson_action(6.0).unwrap();
        let result = metropolis_subgroup_step(
            &mut field,
            [0, 0, 0, 0],
            0,
            6.0,
            proposal,
            0.5,
        )
        .unwrap();
        let after = field.wilson_action(6.0).unwrap();
        assert!(result.accepted);
        assert!((result.delta_action - 0.159_467_377_270_067).abs() < 1e-12);
        assert!((result.acceptance_probability - 0.852_597_781_009_894_1).abs() < 1e-12);
        assert!(((after - before) - result.delta_action).abs() < 1e-12);
    }

    #[test]
    fn rejected_step_restores_original_link_and_action() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        let proposal = Su2SubgroupProposal {
            subgroup: Su2Subgroup::Pair01,
            axis: [1.0, 2.0, 3.0],
            angle: 0.2,
        };
        let original = *field.link([0, 0, 0, 0], 0).unwrap();
        let result = metropolis_subgroup_step(
            &mut field,
            [0, 0, 0, 0],
            0,
            6.0,
            proposal,
            0.99,
        )
        .unwrap();
        assert!(!result.accepted);
        assert!(field.wilson_action(6.0).unwrap().abs() < 1e-14);
        assert_eq!(*field.link([0, 0, 0, 0], 0).unwrap(), original);
    }

    #[test]
    fn local_delta_matches_full_action_on_nontrivial_field() {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        field
            .set_link([0, 0, 0, 0], 1, su3_diagonal(0.13, -0.04))
            .unwrap();
        field
            .set_link([1, 0, 0, 1], 3, su3_diagonal(-0.17, 0.06))
            .unwrap();
        let before = field.wilson_action(5.7).unwrap();
        let result = metropolis_subgroup_step(
            &mut field,
            [0, 1, 0, 1],
            2,
            5.7,
            Su2SubgroupProposal {
                subgroup: Su2Subgroup::Pair02,
                axis: [0.2, 0.4, 0.7],
                angle: -0.11,
            },
            0.0,
        )
        .unwrap();
        let after = field.wilson_action(5.7).unwrap();
        assert!(result.accepted);
        assert!(((after - before) - result.delta_action).abs() < 1e-12);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let bad_axis = Su2SubgroupProposal {
            subgroup: Su2Subgroup::Pair12,
            axis: [0.0, 0.0, 0.0],
            angle: 0.1,
        };
        assert!(matches!(
            metropolis_subgroup_step(&mut field, [0, 0, 0, 0], 0, 6.0, bad_axis, 0.5),
            Err(LatticeMetropolisError::InvalidProposalAxis(_))
        ));
        let proposal = Su2SubgroupProposal {
            subgroup: Su2Subgroup::Pair12,
            axis: [1.0, 0.0, 0.0],
            angle: 0.1,
        };
        assert!(matches!(
            metropolis_subgroup_step(&mut field, [0, 0, 0, 0], 0, 6.0, proposal, 1.0),
            Err(LatticeMetropolisError::InvalidUniformDraw(1.0))
        ));
    }
}
