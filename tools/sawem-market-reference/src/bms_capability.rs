//! Evidence-qualified SOC-dependent battery power capability envelopes.
//!
//! A capability curve is a conservative lower envelope for charge or discharge
//! power over a declared SOC calibration range. Interpolation rounds down so
//! it never invents power between measured/certified knots. Extrapolation is
//! rejected. Synthetic assumptions can be explored but are never labelled
//! operationally qualified.
//!
//! This is a bounded SOC-only envelope. Temperature, SOH, cell imbalance,
//! voltage and dynamic BMS state must be added before it can stand in for a
//! complete battery-management-system capability model.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerDirection {
    Charge,
    Discharge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityEvidenceBasis {
    FieldMeasuredConservative,
    ManufacturerGuaranteed,
    SyntheticAssumption,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityPoint {
    /// State of charge in basis points: 0 = 0%, 10_000 = 100%.
    pub soc_bps: u16,
    /// Conservative maximum AC-side power in watts for this direction.
    pub maximum_power_w: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryCapabilityEnvelope<'a> {
    pub battery_unit_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub rated_power_w: u64,
    pub minimum_soc_bps: u16,
    pub maximum_soc_bps: u16,
    pub evidence_basis: CapabilityEvidenceBasis,
    pub charge_curve: &'a [CapabilityPoint],
    pub discharge_curve: &'a [CapabilityPoint],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityEnvelopeError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidRatedPower,
    InvalidSocRange,
    SocOutsideCalibrationRange,
    InvalidCurve,
    CurveNotStrictlyIncreasing,
    CurveDoesNotCoverDeclaredRange,
    CapabilityAboveRatedPower,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityDisposition {
    WithinEnvelope,
    DerateRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityAssessment<'a> {
    pub battery_unit_id: &'a str,
    pub direction: PowerDirection,
    pub soc_bps: u16,
    pub requested_power_w: u64,
    /// Floor-rounded interpolation of the conservative power envelope.
    pub maximum_supported_power_w: u64,
    pub disposition: CapabilityDisposition,
    /// False for synthetic assumptions. Such a result is a scenario only and
    /// must not be used to authorize a real dispatch.
    pub evidence_basis_is_non_synthetic: bool,
    pub evidence_basis: CapabilityEvidenceBasis,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_curve(
    curve: &[CapabilityPoint],
    minimum_soc_bps: u16,
    maximum_soc_bps: u16,
    rated_power_w: u64,
) -> Result<(), CapabilityEnvelopeError> {
    if curve.len() < 2 {
        return Err(CapabilityEnvelopeError::InvalidCurve);
    }
    if curve.first().map(|point| point.soc_bps) != Some(minimum_soc_bps)
        || curve.last().map(|point| point.soc_bps) != Some(maximum_soc_bps)
    {
        return Err(CapabilityEnvelopeError::CurveDoesNotCoverDeclaredRange);
    }
    for (index, point) in curve.iter().enumerate() {
        if point.soc_bps > 10_000 {
            return Err(CapabilityEnvelopeError::InvalidCurve);
        }
        if point.maximum_power_w > rated_power_w {
            return Err(CapabilityEnvelopeError::CapabilityAboveRatedPower);
        }
        if index > 0 && curve[index - 1].soc_bps >= point.soc_bps {
            return Err(CapabilityEnvelopeError::CurveNotStrictlyIncreasing);
        }
    }
    Ok(())
}

fn interpolate_conservative_power(
    curve: &[CapabilityPoint],
    soc_bps: u16,
) -> Result<u64, CapabilityEnvelopeError> {
    let first = curve.first().ok_or(CapabilityEnvelopeError::InvalidCurve)?;
    let last = curve.last().ok_or(CapabilityEnvelopeError::InvalidCurve)?;
    if soc_bps < first.soc_bps || soc_bps > last.soc_bps {
        return Err(CapabilityEnvelopeError::SocOutsideCalibrationRange);
    }
    if let Some(point) = curve.iter().find(|point| point.soc_bps == soc_bps) {
        return Ok(point.maximum_power_w);
    }

    for pair in curve.windows(2) {
        let lower = pair[0];
        let upper = pair[1];
        if lower.soc_bps < soc_bps && soc_bps < upper.soc_bps {
            let lower_weight = u128::from(upper.soc_bps - soc_bps);
            let upper_weight = u128::from(soc_bps - lower.soc_bps);
            let denominator = u128::from(upper.soc_bps - lower.soc_bps);
            let numerator = u128::from(lower.maximum_power_w)
                .checked_mul(lower_weight)
                .and_then(|value| {
                    u128::from(upper.maximum_power_w)
                        .checked_mul(upper_weight)
                        .and_then(|other| value.checked_add(other))
                })
                .ok_or(CapabilityEnvelopeError::ArithmeticOverflow)?;
            return u64::try_from(numerator / denominator)
                .map_err(|_| CapabilityEnvelopeError::ArithmeticOverflow);
        }
    }
    Err(CapabilityEnvelopeError::SocOutsideCalibrationRange)
}

/// Evaluate a charge/discharge request at a given SOC using a conservative
/// piecewise-linear envelope. No power is inferred outside the evidence range.
pub fn assess_soc_dependent_capability<'a>(
    envelope: BatteryCapabilityEnvelope<'a>,
    direction: PowerDirection,
    soc_bps: u16,
    requested_power_w: u64,
) -> Result<CapabilityAssessment<'a>, CapabilityEnvelopeError> {
    if !nonempty(envelope.battery_unit_id) || !nonempty(envelope.source_ref) {
        return Err(CapabilityEnvelopeError::InvalidEvidenceReference);
    }
    if !valid_sha256(envelope.source_sha256) {
        return Err(CapabilityEnvelopeError::InvalidSourceDigest);
    }
    if envelope.rated_power_w == 0 {
        return Err(CapabilityEnvelopeError::InvalidRatedPower);
    }
    if envelope.minimum_soc_bps >= envelope.maximum_soc_bps
        || envelope.maximum_soc_bps > 10_000
        || soc_bps > 10_000
    {
        return Err(CapabilityEnvelopeError::InvalidSocRange);
    }

    validate_curve(
        envelope.charge_curve,
        envelope.minimum_soc_bps,
        envelope.maximum_soc_bps,
        envelope.rated_power_w,
    )?;
    validate_curve(
        envelope.discharge_curve,
        envelope.minimum_soc_bps,
        envelope.maximum_soc_bps,
        envelope.rated_power_w,
    )?;

    let curve = match direction {
        PowerDirection::Charge => envelope.charge_curve,
        PowerDirection::Discharge => envelope.discharge_curve,
    };
    let maximum_supported_power_w = interpolate_conservative_power(curve, soc_bps)?;

    Ok(CapabilityAssessment {
        battery_unit_id: envelope.battery_unit_id,
        direction,
        soc_bps,
        requested_power_w,
        maximum_supported_power_w,
        disposition: if requested_power_w <= maximum_supported_power_w {
            CapabilityDisposition::WithinEnvelope
        } else {
            CapabilityDisposition::DerateRequired
        },
        evidence_basis_is_non_synthetic: envelope.evidence_basis
            != CapabilityEvidenceBasis::SyntheticAssumption,
        evidence_basis: envelope.evidence_basis,
        source_ref: envelope.source_ref,
        source_sha256: envelope.source_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    const CHARGE: [CapabilityPoint; 3] = [
        CapabilityPoint {
            soc_bps: 1_000,
            maximum_power_w: 2_000,
        },
        CapabilityPoint {
            soc_bps: 5_000,
            maximum_power_w: 4_000,
        },
        CapabilityPoint {
            soc_bps: 9_000,
            maximum_power_w: 1_000,
        },
    ];

    const DISCHARGE: [CapabilityPoint; 3] = [
        CapabilityPoint {
            soc_bps: 1_000,
            maximum_power_w: 1_000,
        },
        CapabilityPoint {
            soc_bps: 5_000,
            maximum_power_w: 4_500,
        },
        CapabilityPoint {
            soc_bps: 9_000,
            maximum_power_w: 3_000,
        },
    ];

    fn envelope<'a>(
        basis: CapabilityEvidenceBasis,
        charge: &'a [CapabilityPoint],
        discharge: &'a [CapabilityPoint],
    ) -> BatteryCapabilityEnvelope<'a> {
        BatteryCapabilityEnvelope {
            battery_unit_id: "LMO-01",
            source_ref: "bms-capability/test-001",
            source_sha256: SOURCE_SHA,
            rated_power_w: 5_000,
            minimum_soc_bps: 1_000,
            maximum_soc_bps: 9_000,
            evidence_basis: basis,
            charge_curve: charge,
            discharge_curve: discharge,
        }
    }

    #[test]
    fn charge_and_discharge_curves_are_direction_specific() {
        let profile = envelope(
            CapabilityEvidenceBasis::FieldMeasuredConservative,
            &CHARGE,
            &DISCHARGE,
        );
        let charge =
            assess_soc_dependent_capability(profile, PowerDirection::Charge, 5_000, 3_500).unwrap();
        let discharge =
            assess_soc_dependent_capability(profile, PowerDirection::Discharge, 5_000, 4_000)
                .unwrap();
        assert_eq!(charge.maximum_supported_power_w, 4_000);
        assert_eq!(charge.disposition, CapabilityDisposition::WithinEnvelope);
        assert_eq!(discharge.maximum_supported_power_w, 4_500);
        assert_eq!(discharge.disposition, CapabilityDisposition::WithinEnvelope);
    }

    #[test]
    fn derating_interpolation_rounds_down_conservatively() {
        let profile = envelope(
            CapabilityEvidenceBasis::FieldMeasuredConservative,
            &CHARGE,
            &DISCHARGE,
        );
        let result =
            assess_soc_dependent_capability(profile, PowerDirection::Charge, 7_001, 2_000).unwrap();
        // Between 50% (4,000 W) and 90% (1,000 W): exact value 2,499.25 W.
        assert_eq!(result.maximum_supported_power_w, 2_499);
        assert_eq!(result.disposition, CapabilityDisposition::WithinEnvelope);
    }

    #[test]
    fn request_above_soc_capability_requires_derating() {
        let profile = envelope(
            CapabilityEvidenceBasis::FieldMeasuredConservative,
            &CHARGE,
            &DISCHARGE,
        );
        let result =
            assess_soc_dependent_capability(profile, PowerDirection::Charge, 9_000, 2_000).unwrap();
        assert_eq!(result.maximum_supported_power_w, 1_000);
        assert_eq!(result.disposition, CapabilityDisposition::DerateRequired);
    }

    #[test]
    fn soc_outside_measured_range_is_rejected_not_extrapolated() {
        let profile = envelope(
            CapabilityEvidenceBasis::FieldMeasuredConservative,
            &CHARGE,
            &DISCHARGE,
        );
        assert_eq!(
            assess_soc_dependent_capability(profile, PowerDirection::Discharge, 9_001, 500),
            Err(CapabilityEnvelopeError::SocOutsideCalibrationRange)
        );
    }

    #[test]
    fn synthetic_assumptions_are_never_evidence_basis_is_non_synthetic() {
        let profile = envelope(
            CapabilityEvidenceBasis::SyntheticAssumption,
            &CHARGE,
            &DISCHARGE,
        );
        let result =
            assess_soc_dependent_capability(profile, PowerDirection::Discharge, 5_000, 1_000)
                .unwrap();
        assert!(!result.evidence_basis_is_non_synthetic);
        assert_eq!(result.disposition, CapabilityDisposition::WithinEnvelope);
    }

    #[test]
    fn invalid_and_unsorted_curves_fail_closed() {
        const UNSORTED: [CapabilityPoint; 3] = [
            CapabilityPoint {
                soc_bps: 1_000,
                maximum_power_w: 2_000,
            },
            CapabilityPoint {
                soc_bps: 7_000,
                maximum_power_w: 3_000,
            },
            CapabilityPoint {
                soc_bps: 5_000,
                maximum_power_w: 4_000,
            },
        ];
        let bad = envelope(
            CapabilityEvidenceBasis::FieldMeasuredConservative,
            &UNSORTED,
            &DISCHARGE,
        );
        assert_eq!(
            assess_soc_dependent_capability(bad, PowerDirection::Charge, 5_000, 1_000),
            Err(CapabilityEnvelopeError::CurveNotStrictlyIncreasing)
        );

        let above_rated = [
            CapabilityPoint {
                soc_bps: 1_000,
                maximum_power_w: 6_000,
            },
            CapabilityPoint {
                soc_bps: 9_000,
                maximum_power_w: 2_000,
            },
        ];
        let bad = envelope(
            CapabilityEvidenceBasis::FieldMeasuredConservative,
            &above_rated,
            &DISCHARGE,
        );
        assert_eq!(
            assess_soc_dependent_capability(bad, PowerDirection::Charge, 5_000, 1_000),
            Err(CapabilityEnvelopeError::CapabilityAboveRatedPower)
        );
    }

    #[test]
    fn malformed_evidence_digest_is_rejected() {
        let profile = BatteryCapabilityEnvelope {
            source_sha256: "not-a-digest",
            ..envelope(
                CapabilityEvidenceBasis::ManufacturerGuaranteed,
                &CHARGE,
                &DISCHARGE,
            )
        };
        assert_eq!(
            assess_soc_dependent_capability(profile, PowerDirection::Charge, 5_000, 1_000),
            Err(CapabilityEnvelopeError::InvalidSourceDigest)
        );
    }
}
