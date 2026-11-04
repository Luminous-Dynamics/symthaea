#![forbid(unsafe_code)]

//! Interval-valued service-deficit arithmetic.
//!
//! This module preserves the distinction between physically measured delivery
//! and delivery qualified for a declared end-use. It evaluates only supplied
//! intervals and qualification states; it does not validate the underlying
//! instruments, assays, installation, or service boundary.

use serde::{Deserialize, Serialize};

use super::EvaluationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RateUnit {
    /// Litres delivered per minute; integrates to litres.
    LitresPerMinute,
    /// Litres delivered per hour; integrates to litres.
    LitresPerHour,
    /// Electrical power input in kW; integrates to kWh-electric.
    ElectricalKilowatts,
    /// Useful thermal power delivered in kW; integrates to kWh-thermal.
    UsefulThermalKilowatts,
}

impl RateUnit {
    const fn seconds_per_unit(self) -> f64 {
        match self {
            Self::LitresPerMinute => 60.0,
            Self::LitresPerHour | Self::ElectricalKilowatts | Self::UsefulThermalKilowatts => {
                3_600.0
            }
        }
    }

    const fn quantity_unit(self) -> QuantityUnit {
        match self {
            Self::LitresPerMinute | Self::LitresPerHour => QuantityUnit::Litres,
            Self::ElectricalKilowatts => QuantityUnit::ElectricalKilowattHours,
            Self::UsefulThermalKilowatts => QuantityUnit::UsefulThermalKilowattHours,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityUnit {
    Litres,
    ElectricalKilowattHours,
    UsefulThermalKilowattHours,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceQualification {
    /// All required qualification gates are declared passing and current.
    /// This state does not authenticate or independently validate them.
    Qualified,
    /// A required gate is known to fail for the declared end-use class.
    KnownFailure,
    /// Qualification is missing, stale, contradictory, or otherwise unresolved.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct RateInterval {
    pub lower: f64,
    pub upper: f64,
    pub unit: RateUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct QuantityInterval {
    pub lower: f64,
    pub upper: f64,
    pub unit: QuantityUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct ServiceDeficitInput {
    /// Requested service rate and its conservative measurement/model bounds.
    pub demand_rate: RateInterval,
    /// Physical delivery rate measured at the declared service boundary.
    pub physical_delivery_rate: RateInterval,
    /// Qualification of the physical delivery for the declared end-use class.
    pub qualification: ServiceQualification,
    /// Positive elapsed duration for this constant/bounded-rate observation block.
    pub duration_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct ServiceDeficitResult {
    pub qualification: ServiceQualification,
    pub duration_seconds: u64,
    /// Requested service quantity over the observation block.
    pub requested: QuantityInterval,
    /// Physically measured quantity, whether or not it qualifies for the end use.
    pub physical_delivered: QuantityInterval,
    /// Quantity credited as qualified service under the declared gate state.
    pub qualified_delivered: QuantityInterval,
    /// Conservative interval for requested minus qualified delivered service.
    pub deficit: QuantityInterval,
}

/// Evaluate a rate-based service deficit without collapsing unknown qualification
/// into either success or a known physical failure.
///
/// For demand [d-, d+] and qualified delivery [q-, q+], the conservative
/// instantaneous deficit rate is [max(0, d- - q+), max(0, d+ - q-)].
/// Interval arithmetic makes no independence or covariance assumption.
///
/// Qualification treatment:
/// - Qualified: use the physical-delivery interval as the qualified interval.
/// - KnownFailure: qualified delivery is [0, 0] for this end-use class.
/// - Unknown: qualified delivery is [0, physical upper bound]. This provides no
///   guaranteed-service credit while preserving a possible upper bound.
///
/// The rate unit must match exactly; this function does not silently convert
/// units. Integrating L/min or L/hour yields litres. Electrical kW yields kWh-electric;
/// useful thermal kW yields kWh-thermal. These energy carriers are not interchangeable.
pub fn evaluate_service_deficit(
    input: ServiceDeficitInput,
) -> Result<ServiceDeficitResult, EvaluationError> {
    validate_rate_interval("service_deficit.demand_rate", input.demand_rate)?;
    validate_rate_interval(
        "service_deficit.physical_delivery_rate",
        input.physical_delivery_rate,
    )?;

    if input.demand_rate.unit != input.physical_delivery_rate.unit {
        return Err(EvaluationError::new(
            "service_deficit.rate_unit",
            "demand and physical delivery rate units must match exactly",
        ));
    }

    if input.duration_seconds == 0 {
        return Err(EvaluationError::new(
            "service_deficit.duration_seconds",
            "observation duration must be positive",
        ));
    }

    let unit = input.demand_rate.unit.quantity_unit();
    let (scale_lower, scale_upper) =
        duration_scale(input.duration_seconds, input.demand_rate.unit)?;

    let requested = scale_interval(
        "service_deficit.requested",
        input.demand_rate.lower,
        input.demand_rate.upper,
        scale_lower,
        scale_upper,
        unit,
    )?;

    let physical_delivered = scale_interval(
        "service_deficit.physical_delivered",
        input.physical_delivery_rate.lower,
        input.physical_delivery_rate.upper,
        scale_lower,
        scale_upper,
        unit,
    )?;

    let qualified_rate = match input.qualification {
        ServiceQualification::Qualified => RateInterval {
            lower: input.physical_delivery_rate.lower,
            upper: input.physical_delivery_rate.upper,
            unit: input.physical_delivery_rate.unit,
        },
        ServiceQualification::KnownFailure => RateInterval {
            lower: 0.0,
            upper: 0.0,
            unit: input.physical_delivery_rate.unit,
        },
        ServiceQualification::Unknown => RateInterval {
            lower: 0.0,
            upper: input.physical_delivery_rate.upper,
            unit: input.physical_delivery_rate.unit,
        },
    };

    let deficit_rate_lower = subtract_rate_lower(input.demand_rate.lower, qualified_rate.upper);
    let deficit_rate_upper = subtract_rate_upper(input.demand_rate.upper, qualified_rate.lower);

    let qualified_delivered = scale_interval(
        "service_deficit.qualified_delivered",
        qualified_rate.lower,
        qualified_rate.upper,
        scale_lower,
        scale_upper,
        unit,
    )?;

    let deficit = scale_interval(
        "service_deficit.deficit",
        deficit_rate_lower,
        deficit_rate_upper,
        scale_lower,
        scale_upper,
        unit,
    )?;

    Ok(ServiceDeficitResult {
        qualification: input.qualification,
        duration_seconds: input.duration_seconds,
        requested,
        physical_delivered,
        qualified_delivered,
        deficit,
    })
}

fn validate_rate_interval(
    path: &'static str,
    interval: RateInterval,
) -> Result<(), EvaluationError> {
    if !interval.lower.is_finite() || !interval.upper.is_finite() {
        return Err(EvaluationError::new(
            path,
            "rate interval bounds must be finite",
        ));
    }
    if interval.lower < 0.0 || interval.upper < 0.0 {
        return Err(EvaluationError::new(
            path,
            "rate interval bounds must be nonnegative",
        ));
    }
    if interval.lower > interval.upper {
        return Err(EvaluationError::new(
            path,
            "rate interval lower bound must not exceed upper bound",
        ));
    }
    Ok(())
}

/// Return the adjacent representable nonnegative f64 below a finite value.
/// Zero is kept exact because all accepted operands are nonnegative.
fn next_down_nonnegative(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

/// Return the adjacent representable f64 above a finite nonnegative value.
fn next_up_nonnegative(value: f64) -> f64 {
    if value == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

/// Widen a positive subtraction outward by one representable step.
fn subtract_rate_lower(left: f64, right: f64) -> f64 {
    let difference = left - right;
    if difference <= 0.0 {
        0.0
    } else {
        next_down_nonnegative(difference).max(0.0)
    }
}

/// Widen a positive subtraction outward by one representable step.
fn subtract_rate_upper(left: f64, right: f64) -> f64 {
    let difference = left - right;
    if difference <= 0.0 {
        0.0
    } else {
        next_up_nonnegative(difference)
    }
}

/// Enclose the exact positive duration/unit ratio despite f64 integer conversion
/// and division rounding. This returns a lower and upper scale interval.
fn duration_scale(duration_seconds: u64, unit: RateUnit) -> Result<(f64, f64), EvaluationError> {
    let duration = duration_seconds as f64;
    let duration_lower = next_down_nonnegative(duration);
    let duration_upper = next_up_nonnegative(duration);
    let divisor = unit.seconds_per_unit();

    let scale_lower = next_down_nonnegative(duration_lower / divisor);
    let scale_upper = next_up_nonnegative(duration_upper / divisor);

    if !scale_lower.is_finite() || !scale_upper.is_finite() || scale_lower <= 0.0 {
        return Err(EvaluationError::new(
            "service_deficit.duration_seconds",
            "duration scaling is not representable as a finite positive interval",
        ));
    }

    Ok((scale_lower, scale_upper))
}

fn scale_interval(
    path: &'static str,
    lower: f64,
    upper: f64,
    scale_lower: f64,
    scale_upper: f64,
    unit: QuantityUnit,
) -> Result<QuantityInterval, EvaluationError> {
    let scaled_lower = lower * scale_lower;
    let scaled_upper = upper * scale_upper;

    if !scaled_lower.is_finite() || !scaled_upper.is_finite() {
        return Err(EvaluationError::new(
            path,
            "rate interval overflowed while integrating over the observation duration",
        ));
    }

    // Exact zero products remain exactly zero; widening them would turn a
    // known-failed end-use gate into a fictitious positive delivery interval.
    let outward_lower = next_down_nonnegative(scaled_lower);
    let outward_upper = if upper == 0.0 || scale_upper == 0.0 {
        // Both factors were exactly zero, so the exact product is zero.
        0.0
    } else {
        // If a positive product underflowed to zero, next_up returns the
        // smallest positive subnormal, retaining a conservative upper bound.
        next_up_nonnegative(scaled_upper)
    };

    if !outward_lower.is_finite() || !outward_upper.is_finite() {
        return Err(EvaluationError::new(
            path,
            "outward-rounded quantity interval is not finite",
        ));
    }

    Ok(QuantityInterval {
        lower: outward_lower,
        upper: outward_upper,
        unit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rate(lower: f64, upper: f64, unit: RateUnit) -> RateInterval {
        RateInterval { lower, upper, unit }
    }

    fn assert_encloses(actual: QuantityInterval, expected_lower: f64, expected_upper: f64) {
        assert!(
            actual.lower <= expected_lower,
            "lower bound {} must enclose expected lower {}",
            actual.lower,
            expected_lower
        );
        assert!(
            actual.upper >= expected_upper,
            "upper bound {} must enclose expected upper {}",
            actual.upper,
            expected_upper
        );
    }

    fn water_input(qualification: ServiceQualification) -> ServiceDeficitInput {
        ServiceDeficitInput {
            demand_rate: rate(1.8, 2.2, RateUnit::LitresPerMinute),
            physical_delivery_rate: rate(1.7, 1.9, RateUnit::LitresPerMinute),
            qualification,
            duration_seconds: 30 * 60,
        }
    }

    #[test]
    fn unknown_quality_preserves_physical_flow_and_bounds_qualified_deficit() {
        let result =
            evaluate_service_deficit(water_input(ServiceQualification::Unknown)).expect("valid");

        assert_encloses(result.physical_delivered, 51.0, 57.0);
        assert_encloses(result.qualified_delivered, 0.0, 57.0);
        assert_encloses(result.deficit, 0.0, 66.0);
        assert_eq!(result.qualification, ServiceQualification::Unknown);
    }

    #[test]
    fn known_quality_failure_receives_no_qualified_service_credit() {
        let result = evaluate_service_deficit(water_input(ServiceQualification::KnownFailure))
            .expect("valid");

        assert_eq!(
            result.qualified_delivered,
            QuantityInterval {
                lower: 0.0,
                upper: 0.0,
                unit: QuantityUnit::Litres,
            }
        );
        assert_encloses(result.deficit, 54.0, 66.0);
        assert_eq!(result.qualification, ServiceQualification::KnownFailure);
    }

    #[test]
    fn qualified_delivery_uses_physical_interval() {
        let result =
            evaluate_service_deficit(water_input(ServiceQualification::Qualified)).expect("valid");

        assert_encloses(result.qualified_delivered, 51.0, 57.0);
        assert_encloses(result.deficit, 0.0, 15.0);
    }

    #[test]
    fn electrical_power_rate_integrates_to_electrical_kwh() {
        let result = evaluate_service_deficit(ServiceDeficitInput {
            demand_rate: rate(1.0, 1.0, RateUnit::ElectricalKilowatts),
            physical_delivery_rate: rate(0.75, 0.75, RateUnit::ElectricalKilowatts),
            qualification: ServiceQualification::Qualified,
            duration_seconds: 2 * 60 * 60,
        })
        .expect("valid");

        assert_encloses(result.deficit, 0.5, 0.5);
    }

    #[test]
    fn useful_thermal_power_integrates_to_thermal_kwh() {
        let result = evaluate_service_deficit(ServiceDeficitInput {
            demand_rate: rate(2.0, 2.0, RateUnit::UsefulThermalKilowatts),
            physical_delivery_rate: rate(1.5, 1.5, RateUnit::UsefulThermalKilowatts),
            qualification: ServiceQualification::Qualified,
            duration_seconds: 2 * 60 * 60,
        })
        .expect("valid");

        assert_eq!(
            result.deficit.unit,
            QuantityUnit::UsefulThermalKilowattHours
        );
        assert_encloses(result.deficit, 1.0, 1.0);
    }

    #[test]
    fn electrical_and_thermal_energy_carriers_cannot_be_mixed() {
        let input = ServiceDeficitInput {
            demand_rate: rate(2.0, 2.0, RateUnit::ElectricalKilowatts),
            physical_delivery_rate: rate(1.0, 1.0, RateUnit::UsefulThermalKilowatts),
            qualification: ServiceQualification::Qualified,
            duration_seconds: 3600,
        };

        let error = evaluate_service_deficit(input).expect_err("energy carriers must not be mixed");
        assert_eq!(error.path, "service_deficit.rate_unit");
    }

    #[test]
    fn litres_per_hour_integrates_over_hours() {
        let result = evaluate_service_deficit(ServiceDeficitInput {
            demand_rate: rate(10.0, 12.0, RateUnit::LitresPerHour),
            physical_delivery_rate: rate(5.0, 6.0, RateUnit::LitresPerHour),
            qualification: ServiceQualification::KnownFailure,
            duration_seconds: 30 * 60,
        })
        .expect("valid");

        assert_encloses(result.deficit, 5.0, 6.0);
    }

    #[test]
    fn mismatched_rate_units_fail_closed() {
        let mut input = water_input(ServiceQualification::Qualified);
        input.physical_delivery_rate.unit = RateUnit::ElectricalKilowatts;

        let error = evaluate_service_deficit(input).expect_err("unit mismatch must fail");
        assert_eq!(error.path, "service_deficit.rate_unit");
    }

    #[test]
    fn non_finite_negative_and_reversed_intervals_fail_closed() {
        let mut input = water_input(ServiceQualification::Qualified);
        input.demand_rate.lower = f64::NAN;
        assert!(evaluate_service_deficit(input).is_err());

        let mut input = water_input(ServiceQualification::Qualified);
        input.physical_delivery_rate.upper = f64::INFINITY;
        assert!(evaluate_service_deficit(input).is_err());

        let mut input = water_input(ServiceQualification::Qualified);
        input.demand_rate.lower = -0.1;
        assert!(evaluate_service_deficit(input).is_err());

        let mut input = water_input(ServiceQualification::Qualified);
        input.demand_rate.lower = 3.0;
        input.demand_rate.upper = 2.0;
        assert!(evaluate_service_deficit(input).is_err());
    }

    #[test]
    fn zero_duration_fails_closed() {
        let mut input = water_input(ServiceQualification::Qualified);
        input.duration_seconds = 0;

        let error = evaluate_service_deficit(input).expect_err("zero duration must fail");
        assert_eq!(error.path, "service_deficit.duration_seconds");
    }

    #[test]
    fn finite_values_that_overflow_on_integration_fail_closed() {
        let input = ServiceDeficitInput {
            demand_rate: rate(f64::MAX, f64::MAX, RateUnit::LitresPerHour),
            physical_delivery_rate: rate(0.0, 0.0, RateUnit::LitresPerHour),
            qualification: ServiceQualification::KnownFailure,
            duration_seconds: u64::MAX,
        };

        let error =
            evaluate_service_deficit(input).expect_err("integrated quantity must remain finite");
        assert_eq!(error.path, "service_deficit.requested");
    }

    #[test]
    fn positive_subnormal_delivery_does_not_underflow_to_exact_zero() {
        let smallest_positive = f64::from_bits(1);
        let result = evaluate_service_deficit(ServiceDeficitInput {
            demand_rate: rate(0.0, smallest_positive, RateUnit::ElectricalKilowatts),
            physical_delivery_rate: rate(0.0, smallest_positive, RateUnit::ElectricalKilowatts),
            qualification: ServiceQualification::Unknown,
            duration_seconds: 1,
        })
        .expect("finite positive subnormal input remains bounded");

        assert_eq!(result.qualified_delivered.lower, 0.0);
        assert!(
            result.qualified_delivered.upper >= smallest_positive,
            "positive physical delivery must not be collapsed to exact zero"
        );
        assert_eq!(result.qualification, ServiceQualification::Unknown);
    }

    #[test]
    fn serialized_result_preserves_unknown_qualification() {
        let result =
            evaluate_service_deficit(water_input(ServiceQualification::Unknown)).expect("valid");
        let json = serde_json::to_string(&result).expect("serialize result");

        assert!(json.contains("\"qualification\":\"unknown\""));
        assert!(json.contains("\"physical_delivered\""));
        assert!(json.contains("\"qualified_delivered\""));
        assert!(json.contains("\"deficit\""));
    }
}
