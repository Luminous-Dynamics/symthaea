// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative validity-horizon and pre-breach handoff decisions.
//!
//! An empirical trend can raise an alarm, but it cannot authorize continued
//! execution. Only a verified growth bound can produce StayQualified.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidityEvidence {
    Missing,
    EmpiricalTrend,
    VerifiedGrowthBound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffDisposition {
    StayQualified,
    PrepareHandoff,
    HandoffNow,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidityHorizonInput {
    pub current_normalized_error_upper: f64,
    pub error_growth_rate_per_time: Option<f64>,
    pub uncertainty_growth_rate_per_time: f64,
    pub tolerance: f64,
    pub remaining_contract_horizon: f64,
    pub handoff_lead_time: f64,
    pub evidence: ValidityEvidence,
    pub physical_gates_pass: bool,
    pub branch_association_stable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidityHorizonAssessment {
    pub disposition: HandoffDisposition,
    /// Estimated time until the normalized error envelope reaches tolerance.
    /// None means the supplied non-negative growth envelope has zero slope.
    pub time_to_threshold: Option<f64>,
    /// For a certified bound, the interval until the next decision must be
    /// recomputed. It never exceeds the remaining declared contract horizon.
    pub next_recheck_after: Option<f64>,
    pub projected_error_at_handoff_completion: Option<f64>,
    pub projected_error_at_contract_end: Option<f64>,
    pub evidence: ValidityEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidityHorizonError {
    NonFiniteInput,
    NegativeErrorOrGrowth,
    InvalidTolerance,
    InvalidContractHorizon,
    InvalidHandoffLeadTime,
    EvidenceRateMismatch,
    ArithmeticOverflow,
}

impl std::fmt::Display for ValidityHorizonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "validity-horizon inputs must be finite"),
            Self::NegativeErrorOrGrowth => write!(f, "error and growth rates must be non-negative"),
            Self::InvalidTolerance => write!(f, "tolerance must be finite and positive"),
            Self::InvalidContractHorizon => {
                write!(f, "remaining contract horizon must be finite and positive")
            }
            Self::InvalidHandoffLeadTime => {
                write!(f, "handoff lead time must be finite and non-negative")
            }
            Self::EvidenceRateMismatch => {
                write!(f, "growth-rate evidence and supplied rate are inconsistent")
            }
            Self::ArithmeticOverflow => write!(f, "validity-horizon arithmetic overflowed"),
        }
    }
}

impl std::error::Error for ValidityHorizonError {}

/// Assess whether a representation can remain in service while a handoff is prepared.
///
/// The error envelope is
/// e_upper(t + tau) = e_upper(t) + (g + u_g) * tau,
/// where g is the error-growth rate and u_g its uncertainty growth rate.
///
/// This function is intentionally fail-closed:
/// - a measured empirical slope can trigger a handoff, but never StayQualified;
/// - Missing evidence returns Unknown;
/// - only VerifiedGrowthBound can authorize StayQualified;
/// - a current tolerance breach, failed physical gate, or unstable branch
///   association returns HandoffNow.
///
/// The supplied "verified" status is a caller assertion backed by external
/// proof evidence (for example a replayable CIC certificate); this function
/// checks arithmetic and decision semantics, not the authenticity of that proof.
pub fn assess_validity_horizon(
    input: ValidityHorizonInput,
) -> Result<ValidityHorizonAssessment, ValidityHorizonError> {
    let scalar_values = [
        input.current_normalized_error_upper,
        input.uncertainty_growth_rate_per_time,
        input.tolerance,
        input.remaining_contract_horizon,
        input.handoff_lead_time,
    ];
    if scalar_values.iter().any(|value| !value.is_finite())
        || input
            .error_growth_rate_per_time
            .is_some_and(|value| !value.is_finite())
    {
        return Err(ValidityHorizonError::NonFiniteInput);
    }
    if input.current_normalized_error_upper < 0.0
        || input.uncertainty_growth_rate_per_time < 0.0
        || input
            .error_growth_rate_per_time
            .is_some_and(|value| value < 0.0)
    {
        return Err(ValidityHorizonError::NegativeErrorOrGrowth);
    }
    if input.tolerance <= 0.0 {
        return Err(ValidityHorizonError::InvalidTolerance);
    }
    if input.remaining_contract_horizon <= 0.0 {
        return Err(ValidityHorizonError::InvalidContractHorizon);
    }
    if input.handoff_lead_time < 0.0 {
        return Err(ValidityHorizonError::InvalidHandoffLeadTime);
    }
    match (input.evidence, input.error_growth_rate_per_time) {
        (ValidityEvidence::Missing, None) => {}
        (ValidityEvidence::Missing, Some(_))
        | (ValidityEvidence::EmpiricalTrend, None)
        | (ValidityEvidence::VerifiedGrowthBound, None) => {
            return Err(ValidityHorizonError::EvidenceRateMismatch);
        }
        (ValidityEvidence::EmpiricalTrend | ValidityEvidence::VerifiedGrowthBound, Some(_)) => {}
    }

    if !input.physical_gates_pass
        || !input.branch_association_stable
        || input.current_normalized_error_upper >= input.tolerance
    {
        return Ok(ValidityHorizonAssessment {
            disposition: HandoffDisposition::HandoffNow,
            time_to_threshold: Some(0.0),
            next_recheck_after: Some(0.0),
            projected_error_at_handoff_completion: Some(input.current_normalized_error_upper),
            projected_error_at_contract_end: Some(input.current_normalized_error_upper),
            evidence: input.evidence,
        });
    }

    let Some(growth_rate) = input.error_growth_rate_per_time else {
        return Ok(ValidityHorizonAssessment {
            disposition: HandoffDisposition::Unknown,
            time_to_threshold: None,
            next_recheck_after: None,
            projected_error_at_handoff_completion: None,
            projected_error_at_contract_end: None,
            evidence: input.evidence,
        });
    };

    let total_growth_rate = growth_rate + input.uncertainty_growth_rate_per_time;
    if !total_growth_rate.is_finite() {
        return Err(ValidityHorizonError::ArithmeticOverflow);
    }
    let projected_error_at_handoff_completion =
        input.current_normalized_error_upper + total_growth_rate * input.handoff_lead_time;
    let projected_error_at_contract_end =
        input.current_normalized_error_upper + total_growth_rate * input.remaining_contract_horizon;
    if !projected_error_at_handoff_completion.is_finite()
        || !projected_error_at_contract_end.is_finite()
    {
        return Err(ValidityHorizonError::ArithmeticOverflow);
    }

    let margin = input.tolerance - input.current_normalized_error_upper;
    let time_to_threshold = if total_growth_rate > 0.0 {
        Some(margin / total_growth_rate)
    } else {
        None
    };
    let within_handoff_window =
        time_to_threshold.is_some_and(|time| time <= input.handoff_lead_time);

    let disposition = match input.evidence {
        ValidityEvidence::Missing => HandoffDisposition::Unknown,
        ValidityEvidence::EmpiricalTrend => {
            if within_handoff_window || projected_error_at_handoff_completion >= input.tolerance {
                HandoffDisposition::PrepareHandoff
            } else {
                HandoffDisposition::Unknown
            }
        }
        ValidityEvidence::VerifiedGrowthBound => {
            if within_handoff_window {
                HandoffDisposition::PrepareHandoff
            } else {
                HandoffDisposition::StayQualified
            }
        }
    };

    let next_recheck_after = match (disposition, time_to_threshold) {
        (HandoffDisposition::StayQualified, Some(time)) => {
            Some((time - input.handoff_lead_time).min(input.remaining_contract_horizon))
        }
        (HandoffDisposition::StayQualified, None) => Some(input.remaining_contract_horizon),
        (HandoffDisposition::PrepareHandoff, _) => Some(0.0),
        _ => None,
    };

    Ok(ValidityHorizonAssessment {
        disposition,
        time_to_threshold,
        next_recheck_after,
        projected_error_at_handoff_completion: Some(projected_error_at_handoff_completion),
        projected_error_at_contract_end: Some(projected_error_at_contract_end),
        evidence: input.evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verified_input() -> ValidityHorizonInput {
        ValidityHorizonInput {
            current_normalized_error_upper: 0.2,
            error_growth_rate_per_time: Some(0.1),
            uncertainty_growth_rate_per_time: 0.02,
            tolerance: 1.0,
            remaining_contract_horizon: 10.0,
            handoff_lead_time: 1.0,
            evidence: ValidityEvidence::VerifiedGrowthBound,
            physical_gates_pass: true,
            branch_association_stable: true,
        }
    }

    #[test]
    fn verified_bound_stays_qualified_until_the_handoff_window() {
        let report = assess_validity_horizon(verified_input()).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::StayQualified);
        assert!((report.time_to_threshold.unwrap() - (0.8 / 0.12)).abs() < 1.0e-12);
        assert!((report.next_recheck_after.unwrap() - (0.8 / 0.12 - 1.0)).abs() < 1.0e-12);
        assert!((report.projected_error_at_handoff_completion.unwrap() - 0.32).abs() < 1.0e-12);
    }

    #[test]
    fn verified_bound_prepares_handoff_before_tolerance_breach() {
        let mut input = verified_input();
        input.current_normalized_error_upper = 0.9;
        input.error_growth_rate_per_time = Some(0.1);
        input.uncertainty_growth_rate_per_time = 0.05;
        input.handoff_lead_time = 1.0;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::PrepareHandoff);
        assert_eq!(report.next_recheck_after, Some(0.0));
        assert!(report.time_to_threshold.unwrap() < input.handoff_lead_time);
    }

    #[test]
    fn empirical_trend_never_authorizes_staying_qualified() {
        let mut input = verified_input();
        input.evidence = ValidityEvidence::EmpiricalTrend;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::Unknown);
    }

    #[test]
    fn empirical_trend_inside_handoff_window_triggers_preparation() {
        let mut input = verified_input();
        input.evidence = ValidityEvidence::EmpiricalTrend;
        input.current_normalized_error_upper = 0.95;
        input.error_growth_rate_per_time = Some(0.1);
        input.uncertainty_growth_rate_per_time = 0.05;
        input.handoff_lead_time = 1.0;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::PrepareHandoff);
    }

    #[test]
    fn missing_growth_evidence_returns_unknown() {
        let mut input = verified_input();
        input.evidence = ValidityEvidence::Missing;
        input.error_growth_rate_per_time = None;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::Unknown);
    }

    #[test]
    fn a_current_tolerance_breach_requires_immediate_handoff() {
        let mut input = verified_input();
        input.current_normalized_error_upper = 1.0;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::HandoffNow);
        assert_eq!(report.time_to_threshold, Some(0.0));
    }

    #[test]
    fn a_failed_physical_gate_requires_immediate_handoff() {
        let mut input = verified_input();
        input.physical_gates_pass = false;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::HandoffNow);
    }

    #[test]
    fn unstable_branch_association_requires_immediate_handoff() {
        let mut input = verified_input();
        input.branch_association_stable = false;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::HandoffNow);
    }

    #[test]
    fn zero_growth_bound_stays_qualified_only_through_contract_horizon() {
        let mut input = verified_input();
        input.error_growth_rate_per_time = Some(0.0);
        input.uncertainty_growth_rate_per_time = 0.0;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::StayQualified);
        assert_eq!(report.time_to_threshold, None);
        assert_eq!(
            report.next_recheck_after,
            Some(input.remaining_contract_horizon)
        );
    }

    #[test]
    fn evidence_and_growth_rate_must_agree() {
        let mut input = verified_input();
        input.evidence = ValidityEvidence::VerifiedGrowthBound;
        input.error_growth_rate_per_time = None;
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::EvidenceRateMismatch)
        );
    }

    #[test]
    fn rejects_non_finite_and_negative_inputs() {
        let mut input = verified_input();
        input.current_normalized_error_upper = f64::NAN;
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::NonFiniteInput)
        );
        let mut input = verified_input();
        input.error_growth_rate_per_time = Some(-0.1);
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::NegativeErrorOrGrowth)
        );
    }

    #[test]
    fn rejects_invalid_horizon_tolerance_and_lead_time() {
        let mut input = verified_input();
        input.tolerance = 0.0;
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::InvalidTolerance)
        );
        let mut input = verified_input();
        input.remaining_contract_horizon = 0.0;
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::InvalidContractHorizon)
        );
        let mut input = verified_input();
        input.handoff_lead_time = -1.0;
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::InvalidHandoffLeadTime)
        );
    }

    #[test]
    fn rejects_growth_arithmetic_overflow() {
        let mut input = verified_input();
        input.error_growth_rate_per_time = Some(f64::MAX);
        input.uncertainty_growth_rate_per_time = f64::MAX;
        assert_eq!(
            assess_validity_horizon(input),
            Err(ValidityHorizonError::ArithmeticOverflow)
        );
    }

    #[test]
    fn recheck_never_exceeds_remaining_contract_horizon() {
        let mut input = verified_input();
        input.remaining_contract_horizon = 2.0;
        input.error_growth_rate_per_time = Some(0.01);
        input.uncertainty_growth_rate_per_time = 0.0;
        let report = assess_validity_horizon(input).unwrap();
        assert_eq!(report.disposition, HandoffDisposition::StayQualified);
        assert_eq!(report.next_recheck_after, Some(2.0));
    }
}
