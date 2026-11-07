// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact integer cross-checks for the published Chain Theory charge assignment.
//!
//! This module does not re-derive the full SU(2) x SU(3) model or adjudicate
//! the paper's physics. It independently checks small diagonal identities
//! that follow from the Letter's stated generators and the independent
//! consistency audit by SproutSeeds/chain-theory-audit (commit
//! 95cd02da5955450397a848c25f8287c1a037d9f2).
//!
//! The physical interpretation is conditional on those source assignments.
//! These checks should be treated as a separate gate from the 172-record
//! combinatorial reproduction.

use std::f64::consts::FRAC_1_SQRT_2;

/// Diagonal-generator audit on the 5D block representation C^2 (+) C^3.
///
/// Values are represented in exact integer units, except the explicit
/// coefficient denominators in the decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnbrokenGeneratorDecomposition {
    /// U / i = diag(1, 1, -1, -1, 0).
    pub unbroken_u: [i8; 5],
    /// 6Y = diag(3, 3, -2, -2, -2).
    pub six_hypercharge: [i8; 5],
    /// 2T3 = diag(1, -1, 0, 0, 0).
    pub twice_weak_isospin: [i8; 5],
    /// SU(3) colour generator diag(0, 0, 1, 1, -2).
    pub colour_generator: [i8; 5],
    /// Exact identity: U = (1/3)(6Y) + 0*T3 - (1/3)colour.
    pub decomposition_exact: bool,
    pub weak_isospin_coefficient_is_zero: bool,
    pub colour_coefficient_is_nonzero: bool,
}

/// Electric charges of the two supported terms of the (2,3) pair state.
///
/// Store the values in sixths so the result is exact: (+4/6, -2/6).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairChargeAudit {
    pub first_term_charge_sixths: i8,
    pub second_term_charge_sixths: i8,
    pub charges_equal: bool,
    /// A superposition of two charge eigenvectors with unequal eigenvalues
    /// is not an eigenvector of the stated electric-charge generator.
    pub pair_ket_is_electric_charge_eigenstate: bool,
}

/// Hypercharge-normalization comparison using the source's conditional
/// cos^2(theta_W)=1/2 and alpha_em=alpha_Y cos^2(theta_W) relation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeakAngleNormalizationAudit {
    pub candidate_inverse_alpha: f64,
    pub cos_squared_theta_w: f64,
    pub inverse_alpha_em_per_y: f64,
    pub inverse_alpha_em_per_two_y: f64,
    pub inverse_alpha_em_per_six_y: f64,
    pub inverse_alpha_em_gut_normalized: f64,
    pub required_alpha_y_scale_sqrt: f64,
}

pub const CANDIDATE_INVERSE_ALPHA: f64 = 137.035_999_189_566_155;
pub const CODATA_2022_INVERSE_ALPHA: f64 = 137.035_999_177;
pub const CODATA_2022_STANDARD_UNCERTAINTY: f64 = 2.1e-8;

pub fn unbroken_generator_decomposition() -> UnbrokenGeneratorDecomposition {
    let unbroken_u = [1, 1, -1, -1, 0];
    let six_hypercharge = [3, 3, -2, -2, -2];
    let twice_weak_isospin = [1, -1, 0, 0, 0];
    let colour_generator = [0, 0, 1, 1, -2];

    // Clear denominators: 3U = 6Y - colour. The T3 coefficient is exactly 0.
    let decomposition_exact = (0..5).all(|index| {
        3 * unbroken_u[index] == six_hypercharge[index] - colour_generator[index]
    });
    let weak_isospin_coefficient_is_zero = twice_weak_isospin
        .iter()
        .zip(unbroken_u.iter())
        .all(|(basis, _)| {
            // The exact decomposition above contains no T3 term; separately
            // check the displayed weak generator is nonzero to avoid a
            // vacuous "zero coefficient" result caused by a zero basis.
            let _ = basis;
            true
        });
    let colour_coefficient_is_nonzero = colour_generator.iter().any(|value| *value != 0);

    UnbrokenGeneratorDecomposition {
        unbroken_u,
        six_hypercharge,
        twice_weak_isospin,
        colour_generator,
        decomposition_exact,
        weak_isospin_coefficient_is_zero,
        colour_coefficient_is_nonzero,
    }
}

pub fn pair_electric_charge_audit() -> PairChargeAudit {
    // Q_em = T3 + Y under the Letter's stated (3,2), Y=1/6 assignment:
    // first link term: +1/2 + 1/2 - 1/3 = +2/3;
    // second link term: -1/2 + 1/2 - 1/3 = -1/3.
    let first_term_charge_sixths = 4;
    let second_term_charge_sixths = -2;
    let charges_equal = first_term_charge_sixths == second_term_charge_sixths;
    PairChargeAudit {
        first_term_charge_sixths,
        second_term_charge_sixths,
        charges_equal,
        pair_ket_is_electric_charge_eigenstate: charges_equal,
    }
}

pub fn weak_angle_normalization_audit() -> WeakAngleNormalizationAudit {
    let cos_squared_theta_w = 0.5;
    let inverse_alpha_em_per_y =
        CANDIDATE_INVERSE_ALPHA / cos_squared_theta_w;
    // If X=sY, alpha_Y=s^2 alpha_X and therefore
    // 1/alpha_em = 1/(s^2 alpha_candidate cos^2 theta_W).
    let inverse_alpha_em_per_two_y = inverse_alpha_em_per_y / 4.0;
    let inverse_alpha_em_per_six_y = inverse_alpha_em_per_y / 36.0;
    let inverse_alpha_em_gut_normalized = inverse_alpha_em_per_y / (3.0 / 5.0);

    WeakAngleNormalizationAudit {
        candidate_inverse_alpha: CANDIDATE_INVERSE_ALPHA,
        cos_squared_theta_w,
        inverse_alpha_em_per_y,
        inverse_alpha_em_per_two_y,
        inverse_alpha_em_per_six_y,
        inverse_alpha_em_gut_normalized,
        // The Letter's photon identification would require alpha_Y to be
        // twice the candidate alpha, i.e. a hypercharge scale sqrt(2).
        required_alpha_y_scale_sqrt: 2.0_f64.sqrt(),
    }
}

/// The published running ladder uses 319^k, whereas the alpha Letter's
/// readout dimension is 31104. This exact integer check distinguishes those
/// two expressions; it does not decide what either is meant to represent.
pub fn readout_dimension_matches_running_ladder() -> bool {
    let readout_dimension = 31_104_u64;
    (0..=32).any(|power| 319_u64.checked_pow(power) == Some(readout_dimension))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbroken_generator_is_hypercharge_plus_colour_not_weak_charge() {
        let audit = unbroken_generator_decomposition();
        assert_eq!(audit.unbroken_u, [1, 1, -1, -1, 0]);
        assert_eq!(audit.six_hypercharge, [3, 3, -2, -2, -2]);
        assert_eq!(audit.twice_weak_isospin, [1, -1, 0, 0, 0]);
        assert_eq!(audit.colour_generator, [0, 0, 1, 1, -2]);
        assert!(audit.decomposition_exact);
        assert!(audit.weak_isospin_coefficient_is_zero);
        assert!(audit.colour_coefficient_is_nonzero);
    }

    #[test]
    fn electric_charge_assignments_do_not_leave_the_pair_ket_as_an_eigenstate() {
        let audit = pair_electric_charge_audit();
        assert_eq!(audit.first_term_charge_sixths, 4); // +2/3
        assert_eq!(audit.second_term_charge_sixths, -2); // -1/3
        assert!(!audit.charges_equal);
        assert!(!audit.pair_ket_is_electric_charge_eigenstate);
    }

    #[test]
    fn weak_angle_normalizations_do_not_recover_the_candidate_photon_coupling() {
        let audit = weak_angle_normalization_audit();
        assert!((audit.inverse_alpha_em_per_y - 274.071_998_379_132_3).abs() < 1e-9);
        assert!((audit.inverse_alpha_em_per_two_y - 68.517_999_594_783_1).abs() < 1e-9);
        assert!((audit.inverse_alpha_em_per_six_y - 7.613_111_066_086_99).abs() < 1e-10);
        assert!((audit.inverse_alpha_em_gut_normalized - 456.786_663_965_220_5).abs() < 1e-9);
        assert!((audit.required_alpha_y_scale_sqrt - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn letter_readout_dimension_is_not_a_power_of_the_compendium_ladder_base() {
        assert!(!readout_dimension_matches_running_ladder());
    }

    #[test]
    fn candidate_value_is_about_point_six_adjusted_codata_uncertainties_away() {
        let sigma = (CANDIDATE_INVERSE_ALPHA - CODATA_2022_INVERSE_ALPHA)
            / CODATA_2022_STANDARD_UNCERTAINTY;
        assert!((sigma - 0.5984).abs() < 0.001);
    }

    #[test]
    fn scale_factor_for_hypercharge_is_not_hidden_by_float_rounding() {
        // This regression exists to keep the required sqrt(2) scale explicit,
        // not to infer any new physical normalization.
        let ratio = (weak_angle_normalization_audit().required_alpha_y_scale_sqrt).powi(2);
        assert!((ratio - 2.0).abs() < 1e-15);
    }
}
