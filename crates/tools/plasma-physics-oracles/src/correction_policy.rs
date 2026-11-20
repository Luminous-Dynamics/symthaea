//! Fail-closed accept/correct/fallback decisions for certified field proposals.
//!
//! This is a routing policy, not a correction solver. It accepts only the
//! exact periodic 1D finite-difference Poisson certificate identity.

#![forbid(unsafe_code)]

use crate::poisson_certificate::{
    PERIODIC_1D_FD_POISSON_OPERATOR_ID, PeriodicPoissonCertificate,
};

/// Action selected for the proposed field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionAction {
    /// Accept the neural proposal because its certified L2 bound meets tolerance.
    AcceptCertifiedProposal,
    /// Spend a correction step and obtain a fresh certificate afterwards.
    RunTrustedCorrection,
    /// Discard the proposal and execute the trusted full solve.
    TrustedFallback,
    /// No safe action is available under the supplied budget and capabilities.
    Unresolved,
}

/// Why the policy selected its action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionReason {
    BoundWithinTolerance,
    BoundExceedsTolerance,
    CertificateMissing,
    CertificateInvalid,
    InvalidTolerance,
    CorrectionBudgetExhausted,
    TrustedFallbackUnavailable,
}

/// Decision record. A caller should persist this together with the solver,
/// task, certificate and execution digests in its execution receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrectionDecision {
    pub action: CorrectionAction,
    pub reason: CorrectionReason,
}

/// Select accept, correction, fallback or unresolved.
///
/// A proposal is accepted only when the certificate is internally consistent,
/// names the supported operator and its field L2 error bound is no greater than
/// the finite positive tolerance. Missing certificates may trigger another
/// trusted correction step; invalid certificates go directly to fallback.
/// Exhausted correction budget never causes implicit acceptance.
pub fn decide_correction(
    certificate: Option<&PeriodicPoissonCertificate>,
    field_l2_tolerance: f64,
    correction_steps_remaining: u32,
    trusted_fallback_available: bool,
) -> CorrectionDecision {
    if !field_l2_tolerance.is_finite() || field_l2_tolerance <= 0.0 {
        return fallback_or_unresolved(
            trusted_fallback_available,
            CorrectionReason::InvalidTolerance,
        );
    }

    match certificate {
        Some(value) if !certificate_is_valid(value) => fallback_or_unresolved(
            trusted_fallback_available,
            CorrectionReason::CertificateInvalid,
        ),
        Some(value) if value.electric_field_error_l2_bound <= field_l2_tolerance => {
            CorrectionDecision {
                action: CorrectionAction::AcceptCertifiedProposal,
                reason: CorrectionReason::BoundWithinTolerance,
            }
        }
        Some(_) if correction_steps_remaining > 0 => CorrectionDecision {
            action: CorrectionAction::RunTrustedCorrection,
            reason: CorrectionReason::BoundExceedsTolerance,
        },
        Some(_) => fallback_or_unresolved(
            trusted_fallback_available,
            CorrectionReason::CorrectionBudgetExhausted,
        ),
        None if correction_steps_remaining > 0 => CorrectionDecision {
            action: CorrectionAction::RunTrustedCorrection,
            reason: CorrectionReason::CertificateMissing,
        },
        None => fallback_or_unresolved(
            trusted_fallback_available,
            CorrectionReason::CertificateMissing,
        ),
    }
}

fn fallback_or_unresolved(
    trusted_fallback_available: bool,
    reason: CorrectionReason,
) -> CorrectionDecision {
    if trusted_fallback_available {
        CorrectionDecision {
            action: CorrectionAction::TrustedFallback,
            reason,
        }
    } else {
        CorrectionDecision {
            action: CorrectionAction::Unresolved,
            reason: CorrectionReason::TrustedFallbackUnavailable,
        }
    }
}

fn certificate_is_valid(value: &PeriodicPoissonCertificate) -> bool {
    if value.operator_id != PERIODIC_1D_FD_POISSON_OPERATOR_ID
        || value.cell_count < 3
        || !value.cell_width.is_finite()
        || value.cell_width <= 0.0
        || !value.lambda_min_nonzero.is_finite()
        || value.lambda_min_nonzero <= 0.0
        || !value.residual_l2.is_finite()
        || value.residual_l2 < 0.0
        || !value.electric_field_error_l2_bound.is_finite()
        || value.electric_field_error_l2_bound < 0.0
        || !value.potential_error_l2_bound.is_finite()
        || value.potential_error_l2_bound < 0.0
        || !value.removed_charge_density_mean.is_finite()
        || !value.removed_potential_mean.is_finite()
        || !value.net_charge_before_projection.is_finite()
    {
        return false;
    }

    let n = value.cell_count as f64;
    let inverse_dx_squared = 1.0 / (value.cell_width * value.cell_width);
    let expected_lambda = 4.0 * (std::f64::consts::PI / n).sin().powi(2) * inverse_dx_squared;
    if !expected_lambda.is_finite() || expected_lambda <= 0.0 {
        return false;
    }

    let lambda_tolerance = 64.0 * f64::EPSILON * expected_lambda.abs();
    if (value.lambda_min_nonzero - expected_lambda).abs() > lambda_tolerance {
        return false;
    }

    let expected_potential_bound = value.residual_l2 / expected_lambda;
    let expected_field_bound = value.residual_l2 / expected_lambda.sqrt();
    let potential_tolerance =
        64.0 * f64::EPSILON * expected_potential_bound.abs().max(f64::MIN_POSITIVE);
    let field_tolerance =
        64.0 * f64::EPSILON * expected_field_bound.abs().max(f64::MIN_POSITIVE);

    (value.potential_error_l2_bound - expected_potential_bound).abs() <= potential_tolerance
        && (value.electric_field_error_l2_bound - expected_field_bound).abs() <= field_tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poisson_certificate::certify_periodic_1d_poisson;
    use std::f64::consts::PI;

    fn certificate_with_bound(target_mode: usize, estimate_mode: usize) -> PeriodicPoissonCertificate {
        let count = 16;
        let dx = 0.5;
        let truth: Vec<f64> = (0..count)
            .map(|cell| (2.0 * PI * target_mode as f64 * cell as f64 / count as f64).cos())
            .collect();
        let estimate: Vec<f64> = (0..count)
            .map(|cell| (2.0 * PI * estimate_mode as f64 * cell as f64 / count as f64).cos())
            .collect();
        let lambda = 4.0 * (PI * target_mode as f64 / count as f64).sin().powi(2) / (dx * dx);
        let charge: Vec<f64> = truth.iter().map(|value| lambda * value).collect();
        certify_periodic_1d_poisson(&estimate, &charge, dx, 1.0, 1.0e-12, 1.0e-12).unwrap()
    }

    #[test]
    fn accepts_only_when_the_certified_field_bound_meets_tolerance() {
        let certificate = certificate_with_bound(1, 1);
        let decision = decide_correction(Some(&certificate), 1.0e-9, 3, true);
        assert_eq!(decision.action, CorrectionAction::AcceptCertifiedProposal);
        assert_eq!(decision.reason, CorrectionReason::BoundWithinTolerance);
    }

    #[test]
    fn small_residual_is_not_enough_when_field_error_bound_exceeds_tolerance() {
        let certificate = certificate_with_bound(1, 2);
        let decision = decide_correction(
            Some(&certificate),
            certificate.electric_field_error_l2_bound / 2.0,
            3,
            true,
        );
        assert_eq!(decision.action, CorrectionAction::RunTrustedCorrection);
        assert_eq!(decision.reason, CorrectionReason::BoundExceedsTolerance);
    }

    #[test]
    fn missing_certificate_spends_budget_but_never_accepts() {
        let decision = decide_correction(None, 1.0e-5, 2, true);
        assert_eq!(decision.action, CorrectionAction::RunTrustedCorrection);
        assert_eq!(decision.reason, CorrectionReason::CertificateMissing);
    }

    #[test]
    fn exhausted_budget_uses_fallback_or_remains_unresolved() {
        let certificate = certificate_with_bound(1, 2);
        let fallback = decide_correction(
            Some(&certificate),
            certificate.electric_field_error_l2_bound / 2.0,
            0,
            true,
        );
        assert_eq!(fallback.action, CorrectionAction::TrustedFallback);
        assert_eq!(fallback.reason, CorrectionReason::CorrectionBudgetExhausted);

        let unresolved = decide_correction(
            Some(&certificate),
            certificate.electric_field_error_l2_bound / 2.0,
            0,
            false,
        );
        assert_eq!(unresolved.action, CorrectionAction::Unresolved);
        assert_eq!(unresolved.reason, CorrectionReason::TrustedFallbackUnavailable);
    }

    #[test]
    fn rejects_wrong_operator_identity() {
        let mut certificate = certificate_with_bound(1, 1);
        certificate.operator_id = "different_stencil_v1";
        let decision = decide_correction(Some(&certificate), 1.0, 10, true);
        assert_eq!(decision.action, CorrectionAction::TrustedFallback);
        assert_eq!(decision.reason, CorrectionReason::CertificateInvalid);
    }

    #[test]
    fn rejects_certificate_with_wrong_discrete_spectral_gap() {
        let mut certificate = certificate_with_bound(1, 1);
        certificate.lambda_min_nonzero *= 0.5;
        certificate.potential_error_l2_bound =
            certificate.residual_l2 / certificate.lambda_min_nonzero;
        certificate.electric_field_error_l2_bound =
            certificate.residual_l2 / certificate.lambda_min_nonzero.sqrt();
        let decision = decide_correction(Some(&certificate), 1.0, 10, true);
        assert_eq!(decision.action, CorrectionAction::TrustedFallback);
        assert_eq!(decision.reason, CorrectionReason::CertificateInvalid);
    }

    #[test]
    fn rejects_mutated_or_non_finite_certificate_bounds() {
        let mut certificate = certificate_with_bound(1, 1);
        certificate.electric_field_error_l2_bound = f64::NAN;
        let decision = decide_correction(Some(&certificate), 1.0, 10, true);
        assert_eq!(decision.action, CorrectionAction::TrustedFallback);
        assert_eq!(decision.reason, CorrectionReason::CertificateInvalid);
    }

    #[test]
    fn invalid_tolerance_fails_closed() {
        let certificate = certificate_with_bound(1, 1);
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let decision = decide_correction(Some(&certificate), tolerance, 10, false);
            assert_eq!(decision.action, CorrectionAction::Unresolved);
            assert_eq!(decision.reason, CorrectionReason::TrustedFallbackUnavailable);
        }
    }
}
