#![forbid(unsafe_code)]

//! Deterministic, offline evaluation primitives for verified household services.
//!
//! This crate evaluates declared evidence and arithmetic only. It does not
//! authenticate source records, inspect physical plumbing, establish water
//! quality, certify a repair, control actuators, or make safety decisions.

pub mod batch;

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluationError {
    pub path: &'static str,
    pub message: String,
}

impl EvaluationError {
    fn new(path: &'static str, message: impl Into<String>) -> Self {
        Self {
            path,
            message: message.into(),
        }
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for EvaluationError {}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct MeterMeasurement {
    pub value_l: f64,
    /// Conservative absolute error bound for this individual measurement term.
    pub absolute_error_bound_l: f64,
    pub stream_id: String,
    pub measurement_boundary_id: String,
    pub window_start_unix_ms: i64,
    pub window_end_unix_ms: i64,
    /// Asserted evidence state; not cryptographic verification.
    pub calibration_qualified: bool,
    /// SHA-256 reference to the calibration record when qualified.
    pub calibration_record_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TopologyAuditReference {
    /// Reference to an independently authored topology-audit report.
    /// The evaluator validates its shape, not its authorship or content.
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WaterBalanceInput {
    pub inlet: MeterMeasurement,
    /// Positive means tank storage increased; negative means it decreased.
    pub tank_storage_delta: MeterMeasurement,
    pub delivered_outlet: MeterMeasurement,
    pub purge_and_overflow: MeterMeasurement,
    pub topology_audit: Option<TopologyAuditReference>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaterBalanceDisposition {
    CalibrationUnresolved,
    TopologyUnresolved,
    /// Zero lies inside the declared residual interval. This is not proof of
    /// physical closure; it only means the declared bound does not exclude it.
    ZeroNotExcludedByDeclaredBound,
    ContradictionBeyondDeclaredBound,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WaterBalanceResult {
    pub residual_l: f64,
    pub residual_lower_l: f64,
    pub residual_upper_l: f64,
    pub aggregate_absolute_error_bound_l: f64,
    pub disposition: WaterBalanceDisposition,
    pub topology_audit_reference_present: bool,
    pub uncertainty_semantics: String,
}

/// Recompute inlet - tank storage delta - delivered outlet - purge/overflow.
///
/// A result inside the declared uncertainty interval is not labelled a pass.
/// The topology-audit digest is only a reference; the caller remains
/// responsible for verifying the report and auditor authority independently.
pub fn evaluate_water_balance(
    input: &WaterBalanceInput,
) -> Result<WaterBalanceResult, EvaluationError> {
    let terms = [
        ("inlet", &input.inlet),
        ("tank_storage_delta", &input.tank_storage_delta),
        ("delivered_outlet", &input.delivered_outlet),
        ("purge_and_overflow", &input.purge_and_overflow),
    ];

    let mut streams = BTreeSet::new();
    let mut boundaries = BTreeSet::new();
    let expected_window = (
        input.inlet.window_start_unix_ms,
        input.inlet.window_end_unix_ms,
    );

    if expected_window.0 >= expected_window.1 {
        return Err(EvaluationError::new(
            "window",
            "observation window must have positive duration",
        ));
    }

    let mut all_calibrated = true;
    let mut aggregate_absolute_error_bound_l = 0.0;
    for (name, measurement) in terms {
        if !measurement.value_l.is_finite() {
            return Err(EvaluationError::new(
                name,
                "measurement value must be finite",
            ));
        }
        if !measurement.absolute_error_bound_l.is_finite()
            || measurement.absolute_error_bound_l < 0.0
        {
            return Err(EvaluationError::new(
                name,
                "per-measurement absolute error bound must be finite and nonnegative",
            ));
        }
        aggregate_absolute_error_bound_l += measurement.absolute_error_bound_l;
        if !aggregate_absolute_error_bound_l.is_finite() {
            return Err(EvaluationError::new(
                "aggregate_absolute_error_bound_l",
                "sum of per-measurement error bounds is not finite",
            ));
        }
        if measurement.stream_id.trim().is_empty() {
            return Err(EvaluationError::new(
                name,
                "stream identifier must not be empty",
            ));
        }
        if measurement.measurement_boundary_id.trim().is_empty() {
            return Err(EvaluationError::new(
                name,
                "measurement boundary identifier must not be empty",
            ));
        }
        if (
            measurement.window_start_unix_ms,
            measurement.window_end_unix_ms,
        ) != expected_window
        {
            return Err(EvaluationError::new(
                name,
                "all four measurements must cover the same observation window",
            ));
        }
        if !streams.insert(measurement.stream_id.as_str()) {
            return Err(EvaluationError::new(
                name,
                "measurement stream identifiers must be unique",
            ));
        }
        if !boundaries.insert(measurement.measurement_boundary_id.as_str()) {
            return Err(EvaluationError::new(
                name,
                "measurement boundary identifiers must be unique",
            ));
        }
        if measurement.calibration_qualified {
            let digest = measurement
                .calibration_record_sha256
                .as_deref()
                .ok_or_else(|| {
                    EvaluationError::new(
                        name,
                        "qualified calibration requires a record digest reference",
                    )
                })?;
            validate_sha256(digest, name)?;
        } else {
            all_calibrated = false;
        }
    }

    for (name, measurement) in [
        ("inlet", &input.inlet),
        ("delivered_outlet", &input.delivered_outlet),
        ("purge_and_overflow", &input.purge_and_overflow),
    ] {
        if measurement.value_l < 0.0 {
            return Err(EvaluationError::new(
                name,
                "volume magnitude must not be negative",
            ));
        }
    }

    if let Some(audit) = &input.topology_audit {
        validate_sha256(&audit.report_sha256, "topology_audit.report_sha256")?;
    }

    let residual_l = input.inlet.value_l
        - input.tank_storage_delta.value_l
        - input.delivered_outlet.value_l
        - input.purge_and_overflow.value_l;
    if !residual_l.is_finite() {
        return Err(EvaluationError::new(
            "residual_l",
            "computed residual is not finite",
        ));
    }

    let residual_lower_l = residual_l - aggregate_absolute_error_bound_l;
    let residual_upper_l = residual_l + aggregate_absolute_error_bound_l;
    if !residual_lower_l.is_finite() || !residual_upper_l.is_finite() {
        return Err(EvaluationError::new(
            "residual_interval",
            "computed residual interval is not finite",
        ));
    }

    let disposition = if !all_calibrated {
        WaterBalanceDisposition::CalibrationUnresolved
    } else if input.topology_audit.is_none() {
        WaterBalanceDisposition::TopologyUnresolved
    } else if residual_lower_l <= 0.0 && residual_upper_l >= 0.0 {
        WaterBalanceDisposition::ZeroNotExcludedByDeclaredBound
    } else {
        WaterBalanceDisposition::ContradictionBeyondDeclaredBound
    };

    Ok(WaterBalanceResult {
        residual_l,
        residual_lower_l,
        residual_upper_l,
        aggregate_absolute_error_bound_l,
        disposition,
        topology_audit_reference_present: input.topology_audit.is_some(),
        uncertainty_semantics: "worst-case interval propagation by summing per-term absolute error bounds; no independence or covariance assumption".to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ReserveEnvelopeInput {
    /// Usable volume at the declared delivery boundary, not gross tank level.
    pub usable_volume_lower_l: f64,
    pub usable_volume_upper_l: f64,
    /// Bounded household demand for the declared service profile.
    pub demand_lower_l_per_hour: f64,
    pub demand_upper_l_per_hour: f64,
    /// Minimum pressure observed at the actual delivery boundary.
    pub minimum_observed_pressure_kpa: f64,
    pub required_minimum_pressure_kpa: f64,
    pub pump_available: bool,
    pub electrical_path_available: bool,
    pub delivery_flow_pressure_qualified: bool,
    /// Separate independent qualification for the specific intended end use.
    pub end_use_quality_qualified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReserveDisposition {
    NoDeliveryPath,
    DeliveryEnvelopeUnqualified,
    EndUseNotQualified,
    DurationRangeAvailable,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ReserveEstimate {
    pub disposition: ReserveDisposition,
    pub lower_bound_hours: Option<f64>,
    pub upper_bound_hours: Option<f64>,
    pub functional_unit: String,
}

/// Estimate a conservative service-duration range.
///
/// Inputs must already be qualified for the declared use and measured at the
/// service-delivery boundary. Gross tank volume is deliberately not accepted
/// as a separate input. The result is a conditional estimate, not a forecast.
pub fn estimate_deliverable_service_hours(
    input: &ReserveEnvelopeInput,
) -> Result<ReserveEstimate, EvaluationError> {
    for (name, value) in [
        ("usable_volume_lower_l", input.usable_volume_lower_l),
        ("usable_volume_upper_l", input.usable_volume_upper_l),
        ("demand_lower_l_per_hour", input.demand_lower_l_per_hour),
        ("demand_upper_l_per_hour", input.demand_upper_l_per_hour),
        (
            "minimum_observed_pressure_kpa",
            input.minimum_observed_pressure_kpa,
        ),
        (
            "required_minimum_pressure_kpa",
            input.required_minimum_pressure_kpa,
        ),
    ] {
        if !value.is_finite() {
            return Err(EvaluationError::new(
                "reserve_envelope",
                format!("{name} must be finite"),
            ));
        }
    }

    if input.usable_volume_lower_l < 0.0
        || input.usable_volume_upper_l < input.usable_volume_lower_l
    {
        return Err(EvaluationError::new(
            "usable_volume",
            "volume interval must be nonnegative and ordered",
        ));
    }
    if input.demand_lower_l_per_hour <= 0.0
        || input.demand_upper_l_per_hour < input.demand_lower_l_per_hour
    {
        return Err(EvaluationError::new(
            "demand",
            "demand interval must be positive and ordered",
        ));
    }
    if input.minimum_observed_pressure_kpa < 0.0 || input.required_minimum_pressure_kpa <= 0.0 {
        return Err(EvaluationError::new(
            "pressure",
            "observed pressure must be nonnegative and required pressure positive",
        ));
    }

    let disposition = if !input.pump_available || !input.electrical_path_available {
        ReserveDisposition::NoDeliveryPath
    } else if !input.delivery_flow_pressure_qualified
        || input.minimum_observed_pressure_kpa < input.required_minimum_pressure_kpa
    {
        ReserveDisposition::DeliveryEnvelopeUnqualified
    } else if !input.end_use_quality_qualified {
        ReserveDisposition::EndUseNotQualified
    } else {
        ReserveDisposition::DurationRangeAvailable
    };

    if disposition != ReserveDisposition::DurationRangeAvailable {
        return Ok(ReserveEstimate {
            disposition,
            lower_bound_hours: None,
            upper_bound_hours: None,
            functional_unit: "hours of the declared qualified household service".to_owned(),
        });
    }

    let lower = input.usable_volume_lower_l / input.demand_upper_l_per_hour;
    let upper = input.usable_volume_upper_l / input.demand_lower_l_per_hour;
    if !lower.is_finite() || !upper.is_finite() {
        return Err(EvaluationError::new(
            "duration_range",
            "computed duration is not finite",
        ));
    }

    Ok(ReserveEstimate {
        disposition,
        lower_bound_hours: Some(lower),
        upper_bound_hours: Some(upper),
        functional_unit: "hours of the declared qualified household service".to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RepairAttempt {
    pub diagnosis_supported: bool,
    pub exact_spare_available: bool,
    pub replacement_compatibility_qualified: bool,
    pub offline_service_required: bool,
    pub offline_service_available: bool,
    pub affected_calibration_requalified: bool,
    pub post_repair_function_passed: bool,
    pub safety_check_required: bool,
    pub post_repair_safety_check_passed: bool,
    pub follow_up_days_observed: u32,
    pub recurrence_observed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairDisposition {
    Blocked,
    HeldForRequalification,
    RecurrenceObserved,
    InitiallyRestoredPendingFollowUp,
    NoRecurrenceObservedDuringFollowUp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairEvaluation {
    pub disposition: RepairDisposition,
    pub follow_up_days_observed: u32,
    pub target_follow_up_days: u32,
    pub lifetime_reliability_claim_permitted: bool,
}

/// Evaluate repair evidence without turning a successful restart into a
/// durability or lifetime claim.
pub fn evaluate_repair_attempt(
    input: &RepairAttempt,
    target_follow_up_days: u32,
) -> Result<RepairEvaluation, EvaluationError> {
    if target_follow_up_days == 0 {
        return Err(EvaluationError::new(
            "target_follow_up_days",
            "target follow-up must be greater than zero",
        ));
    }

    let disposition = if !input.diagnosis_supported
        || !input.exact_spare_available
        || !input.replacement_compatibility_qualified
        || (input.offline_service_required && !input.offline_service_available)
    {
        RepairDisposition::Blocked
    } else if !input.affected_calibration_requalified
        || !input.post_repair_function_passed
        || (input.safety_check_required && !input.post_repair_safety_check_passed)
    {
        RepairDisposition::HeldForRequalification
    } else if input.recurrence_observed {
        RepairDisposition::RecurrenceObserved
    } else if input.follow_up_days_observed >= target_follow_up_days {
        RepairDisposition::NoRecurrenceObservedDuringFollowUp
    } else {
        RepairDisposition::InitiallyRestoredPendingFollowUp
    };

    Ok(RepairEvaluation {
        disposition,
        follow_up_days_observed: input.follow_up_days_observed,
        target_follow_up_days,
        lifetime_reliability_claim_permitted: false,
    })
}

fn validate_sha256(value: &str, path: &'static str) -> Result<(), EvaluationError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EvaluationError::new(
            path,
            "expected a 64-character hexadecimal SHA-256 reference",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(
        value_l: f64,
        stream: &str,
        boundary: &str,
        start: i64,
        end: i64,
    ) -> MeterMeasurement {
        MeterMeasurement {
            value_l,
            stream_id: stream.to_owned(),
            measurement_boundary_id: boundary.to_owned(),
            window_start_unix_ms: start,
            window_end_unix_ms: end,
            absolute_error_bound_l: 1.25,
            calibration_qualified: true,
            calibration_record_sha256: Some("a".repeat(64)),
        }
    }

    fn balance() -> WaterBalanceInput {
        WaterBalanceInput {
            inlet: measurement(480.0, "inlet-stream", "inlet-boundary", 1, 1001),
            tank_storage_delta: measurement(120.0, "tank-stream", "tank-boundary", 1, 1001),
            delivered_outlet: measurement(520.0, "outlet-stream", "outlet-boundary", 1, 1001),
            purge_and_overflow: measurement(40.0, "purge-stream", "purge-boundary", 1, 1001),
            topology_audit: Some(TopologyAuditReference {
                report_sha256: "b".repeat(64),
            }),
        }
    }

    fn repair() -> RepairAttempt {
        RepairAttempt {
            diagnosis_supported: true,
            exact_spare_available: true,
            replacement_compatibility_qualified: true,
            offline_service_required: true,
            offline_service_available: true,
            affected_calibration_requalified: true,
            post_repair_function_passed: true,
            safety_check_required: true,
            post_repair_safety_check_passed: true,
            follow_up_days_observed: 30,
            recurrence_observed: false,
        }
    }

    #[test]
    fn signed_balance_recomputes_wcont_008_to_negative_two_hundred_litres() {
        let result = evaluate_water_balance(&balance()).unwrap();
        assert_eq!(result.residual_l, -200.0);
        assert_eq!(result.residual_lower_l, -205.0);
        assert_eq!(result.residual_upper_l, -195.0);
        assert_eq!(result.aggregate_absolute_error_bound_l, 5.0);
        assert_eq!(
            result.disposition,
            WaterBalanceDisposition::ContradictionBeyondDeclaredBound
        );
    }

    #[test]
    fn tank_storage_increase_is_subtracted_from_inlet() {
        let mut input = balance();
        input.inlet.value_l = 200.0;
        input.tank_storage_delta.value_l = 100.0;
        input.delivered_outlet.value_l = 90.0;
        input.purge_and_overflow.value_l = 10.0;
        assert_eq!(evaluate_water_balance(&input).unwrap().residual_l, 0.0);
    }

    #[test]
    fn negative_storage_delta_represents_a_drawdown() {
        let mut input = balance();
        input.inlet.value_l = 100.0;
        input.tank_storage_delta.value_l = -50.0;
        input.delivered_outlet.value_l = 140.0;
        input.purge_and_overflow.value_l = 10.0;
        assert_eq!(evaluate_water_balance(&input).unwrap().residual_l, 0.0);
    }

    #[test]
    fn residual_inside_bound_is_not_called_a_pass() {
        let mut input = balance();
        input.inlet.value_l = 480.0;
        input.tank_storage_delta.value_l = 0.0;
        input.delivered_outlet.value_l = 475.0;
        input.purge_and_overflow.value_l = 0.0;
        assert_eq!(
            evaluate_water_balance(&input).unwrap().disposition,
            WaterBalanceDisposition::ZeroNotExcludedByDeclaredBound
        );
    }

    #[test]
    fn missing_topology_reference_blocks_topology_qualified_disposition() {
        let mut input = balance();
        input.topology_audit = None;
        assert_eq!(
            evaluate_water_balance(&input).unwrap().disposition,
            WaterBalanceDisposition::TopologyUnresolved
        );
    }

    #[test]
    fn unqualified_calibration_blocks_balance_claim() {
        let mut input = balance();
        input.delivered_outlet.calibration_qualified = false;
        input.delivered_outlet.calibration_record_sha256 = None;
        assert_eq!(
            evaluate_water_balance(&input).unwrap().disposition,
            WaterBalanceDisposition::CalibrationUnresolved
        );
    }

    #[test]
    fn mismatched_windows_are_rejected() {
        let mut input = balance();
        input.purge_and_overflow.window_end_unix_ms = 999;
        assert!(evaluate_water_balance(&input).is_err());
    }

    #[test]
    fn duplicate_boundaries_are_rejected_even_with_different_instruments() {
        let mut input = balance();
        input.delivered_outlet.measurement_boundary_id = "inlet-boundary".to_owned();
        assert!(evaluate_water_balance(&input).is_err());
    }

    #[test]
    fn negative_uncertainty_is_rejected() {
        let mut input = balance();
        input.inlet.absolute_error_bound_l = -0.1;
        assert!(evaluate_water_balance(&input).is_err());
    }

    #[test]
    fn non_finite_per_measurement_uncertainty_is_rejected() {
        let mut input = balance();
        input.inlet.absolute_error_bound_l = f64::INFINITY;
        assert!(evaluate_water_balance(&input).is_err());
    }

    #[test]
    fn aggregate_uncertainty_is_the_conservative_sum_of_all_four_terms() {
        let mut input = balance();
        input.inlet.absolute_error_bound_l = 1.0;
        input.tank_storage_delta.absolute_error_bound_l = 2.0;
        input.delivered_outlet.absolute_error_bound_l = 3.0;
        input.purge_and_overflow.absolute_error_bound_l = 4.0;
        let result = evaluate_water_balance(&input).unwrap();
        assert_eq!(result.aggregate_absolute_error_bound_l, 10.0);
        assert_eq!(result.residual_lower_l, -210.0);
        assert_eq!(result.residual_upper_l, -190.0);
    }

    #[test]
    fn overflow_when_summing_measurement_bounds_is_rejected() {
        let mut input = balance();
        input.inlet.absolute_error_bound_l = f64::MAX;
        input.tank_storage_delta.absolute_error_bound_l = f64::MAX;
        assert!(evaluate_water_balance(&input).is_err());
    }

    #[test]
    fn non_finite_meter_values_are_rejected() {
        let mut input = balance();
        input.inlet.value_l = f64::NAN;
        assert!(evaluate_water_balance(&input).is_err());
    }

    fn reserve() -> ReserveEnvelopeInput {
        ReserveEnvelopeInput {
            usable_volume_lower_l: 200.0,
            usable_volume_upper_l: 250.0,
            demand_lower_l_per_hour: 10.0,
            demand_upper_l_per_hour: 20.0,
            minimum_observed_pressure_kpa: 160.0,
            required_minimum_pressure_kpa: 150.0,
            pump_available: true,
            electrical_path_available: true,
            delivery_flow_pressure_qualified: true,
            end_use_quality_qualified: true,
        }
    }

    #[test]
    fn unavailable_pump_or_power_never_produces_service_hours() {
        let mut input = reserve();
        input.pump_available = false;
        let result = estimate_deliverable_service_hours(&input).unwrap();
        assert_eq!(result.disposition, ReserveDisposition::NoDeliveryPath);
        assert!(result.lower_bound_hours.is_none());
    }

    #[test]
    fn service_hours_use_deliverable_volume_and_bounded_demand() {
        let result = estimate_deliverable_service_hours(&reserve()).unwrap();
        assert_eq!(
            result.disposition,
            ReserveDisposition::DurationRangeAvailable
        );
        assert_eq!(result.lower_bound_hours, Some(10.0));
        assert_eq!(result.upper_bound_hours, Some(25.0));
    }

    #[test]
    fn storage_quantity_without_end_use_quality_does_not_make_service() {
        let mut input = reserve();
        input.end_use_quality_qualified = false;
        let result = estimate_deliverable_service_hours(&input).unwrap();
        assert_eq!(result.disposition, ReserveDisposition::EndUseNotQualified);
        assert!(result.upper_bound_hours.is_none());
    }

    #[test]
    fn low_delivery_pressure_blocks_service_duration_estimate() {
        let mut input = reserve();
        input.minimum_observed_pressure_kpa = 90.0;
        let result = estimate_deliverable_service_hours(&input).unwrap();
        assert_eq!(
            result.disposition,
            ReserveDisposition::DeliveryEnvelopeUnqualified
        );
    }

    #[test]
    fn repair_without_post_repair_safety_test_is_held() {
        let mut input = repair();
        input.post_repair_safety_check_passed = false;
        assert_eq!(
            evaluate_repair_attempt(&input, 30).unwrap().disposition,
            RepairDisposition::HeldForRequalification
        );
    }

    #[test]
    fn repair_with_recurrence_is_not_durable_success() {
        let mut input = repair();
        input.recurrence_observed = true;
        assert_eq!(
            evaluate_repair_attempt(&input, 30).unwrap().disposition,
            RepairDisposition::RecurrenceObserved
        );
        assert!(
            !evaluate_repair_attempt(&input, 30)
                .unwrap()
                .lifetime_reliability_claim_permitted
        );
    }

    #[test]
    fn initial_restore_is_distinct_from_followup_completion() {
        let mut input = repair();
        input.follow_up_days_observed = 2;
        assert_eq!(
            evaluate_repair_attempt(&input, 30).unwrap().disposition,
            RepairDisposition::InitiallyRestoredPendingFollowUp
        );
    }

    #[test]
    fn followup_only_supports_a_bounded_no_recurrence_statement() {
        let result = evaluate_repair_attempt(&repair(), 30).unwrap();
        assert_eq!(
            result.disposition,
            RepairDisposition::NoRecurrenceObservedDuringFollowUp
        );
        assert_eq!(result.follow_up_days_observed, 30);
        assert!(!result.lifetime_reliability_claim_permitted);
    }

    #[test]
    fn cloud_only_service_blocks_an_offline_repair_profile() {
        let mut input = repair();
        input.offline_service_available = false;
        assert_eq!(
            evaluate_repair_attempt(&input, 30).unwrap().disposition,
            RepairDisposition::Blocked
        );
    }

    #[test]
    fn zero_followup_target_is_rejected() {
        assert!(evaluate_repair_attempt(&repair(), 0).is_err());
    }
}
