#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative control-chart screening for a predeclared metrology run.
//!
//! This pure Rust core classifies the scheduled control readings, then marks DUT
//! observations in the uncertainty interval surrounding an out-of-band, missing,
//! or invalid control as NOT_EVALUATED. It does not estimate/correct drift and
//! does not turn "no excursion detected" into proof of measurement accuracy.

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledControl {
    /// Stable, unique control slot identity.
    pub id: String,
    /// Expected slot position in monotonic elapsed seconds. For missing readings,
    /// this planned time is used as the conservative interval anchor.
    pub time_seconds: f64,
    /// The measured control value, or None if the scheduled control was missed.
    pub value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DutObservation {
    pub id: String,
    /// Monotonic elapsed seconds in the same time base as control slots.
    pub time_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftControlPlan {
    pub run_start_seconds: f64,
    pub run_end_seconds: f64,
    pub nominal_value: f64,
    /// Absolute half-width of the predeclared guard band. Boundaries are inclusive.
    pub guard_half_width: f64,
    /// Predeclared upper bound for planned spacing, including run-edge gaps.
    /// This is an experimental design constraint, not a validated drift-detection rate.
    pub maximum_control_gap_seconds: f64,
    pub controls: Vec<ScheduledControl>,
    pub dut_observations: Vec<DutObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlDisposition {
    InBand,
    OutOfBand,
    Missing,
    Invalid,
}

impl ControlDisposition {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InBand => "IN_BAND",
            Self::OutOfBand => "OUT_OF_BAND",
            Self::Missing => "MISSING",
            Self::Invalid => "INVALID",
        }
    }

    const fn is_acceptable(self) -> bool {
        matches!(self, Self::InBand)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ControlDecision {
    pub control_id: String,
    pub time_seconds: f64,
    pub measured_value: Option<f64>,
    pub disposition: ControlDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationDisposition {
    NoOutOfBandControlDetected,
    QuarantinedByControlExcursion,
}

impl ObservationDisposition {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoOutOfBandControlDetected => "NO_OUT_OF_BAND_CONTROL_DETECTED",
            Self::QuarantinedByControlExcursion => "NOT_EVALUATED_CONTROL_EXCURSION",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationDecision {
    pub observation_id: String,
    pub time_seconds: f64,
    pub disposition: ObservationDisposition,
    /// IDs of control slots responsible for quarantine, in schedule order.
    pub responsible_control_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftScreeningReport {
    pub guard_lower: f64,
    pub guard_upper: f64,
    pub maximum_control_gap_seconds: f64,
    pub scheduled_max_control_gap_seconds: f64,
    pub controls: Vec<ControlDecision>,
    pub observations: Vec<ObservationDecision>,
    /// Hard boundary for interpretation: this report is not a physical qualification.
    pub qualification: &'static str,
}

/// Apply a frozen absolute guard band and conservative quarantine intervals.
///
/// For every missing, invalid, or out-of-band control slot, its quarantine window
/// extends from the preceding acceptable control (or run start) through the next
/// acceptable control (or run end), inclusive. Overlapping windows are retained as
/// their responsible slot IDs rather than silently averaged or normalized away.
///
/// A "NO_OUT_OF_BAND_CONTROL_DETECTED" observation means only that no bad scheduled
/// control window brackets it under this algorithm. It is not proof of calibration,
/// traceability, stability between controls, or measurement fitness.
///
/// # Errors
///
/// Rejects invalid run bounds/configuration, duplicate/empty IDs, unordered control
/// slots, out-of-run timestamps, and invalid DUT-observation timestamps.
pub fn screen_drift_controls(plan: &DriftControlPlan) -> Result<DriftScreeningReport, Vec<String>> {
    let mut errors = Vec::new();

    if !plan.run_start_seconds.is_finite()
        || !plan.run_end_seconds.is_finite()
        || plan.run_end_seconds <= plan.run_start_seconds
    {
        errors.push("run bounds must be finite and end after start".to_owned());
    }
    if !plan.nominal_value.is_finite() {
        errors.push("control nominal value must be finite".to_owned());
    }
    if !plan.guard_half_width.is_finite() || plan.guard_half_width < 0.0 {
        errors.push("guard half-width must be finite and nonnegative".to_owned());
    }
    if !plan.maximum_control_gap_seconds.is_finite()
        || plan.maximum_control_gap_seconds <= 0.0
    {
        errors.push("maximum control gap must be finite and positive".to_owned());
    }

    let lower = plan.nominal_value - plan.guard_half_width;
    let upper = plan.nominal_value + plan.guard_half_width;
    if !lower.is_finite() || !upper.is_finite() {
        errors.push("control guard-band bounds must be finite".to_owned());
    }

    let mut control_ids = BTreeSet::new();
    let mut previous_time: Option<f64> = None;
    for control in &plan.controls {
        if control.id.trim().is_empty() {
            errors.push("control slot IDs must be nonempty".to_owned());
        }
        if !control_ids.insert(control.id.clone()) {
            errors.push(format!("duplicate control slot ID: {}", control.id));
        }
        if !control.time_seconds.is_finite()
            || control.time_seconds < plan.run_start_seconds
            || control.time_seconds > plan.run_end_seconds
        {
            errors.push(format!(
                "control slot {} timestamp is non-finite or outside the run",
                control.id
            ));
        }
        if previous_time.is_some_and(|previous| control.time_seconds <= previous) {
            errors.push(format!(
                "control schedule is not strictly time-ordered at {}",
                control.id
            ));
        }
        previous_time = Some(control.time_seconds);
    }

    let mut observation_ids = BTreeSet::new();
    for observation in &plan.dut_observations {
        if observation.id.trim().is_empty() {
            errors.push("DUT observation IDs must be nonempty".to_owned());
        }
        if !observation_ids.insert(observation.id.clone()) {
            errors.push(format!("duplicate DUT observation ID: {}", observation.id));
        }
        if !observation.time_seconds.is_finite()
            || observation.time_seconds < plan.run_start_seconds
            || observation.time_seconds > plan.run_end_seconds
        {
            errors.push(format!(
                "DUT observation {} timestamp is non-finite or outside the run",
                observation.id
            ));
        }
    }

    if plan.controls.is_empty() {
        errors.push("at least one predeclared control slot is required".to_owned());
    }
    if !errors.is_empty() {
        errors.sort();
        errors.dedup();
        return Err(errors);
    }

    // Include the run edges so a schedule with dense internal controls cannot hide
    // an unmonitored start or end interval.
    let mut scheduled_times = Vec::with_capacity(plan.controls.len() + 2);
    scheduled_times.push(plan.run_start_seconds);
    scheduled_times.extend(plan.controls.iter().map(|control| control.time_seconds));
    scheduled_times.push(plan.run_end_seconds);
    let scheduled_max_control_gap_seconds = scheduled_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(0.0_f64, f64::max);
    if scheduled_max_control_gap_seconds > plan.maximum_control_gap_seconds {
        return Err(vec![format!(
            "scheduled control gap exceeds predeclared maximum: {} seconds, allowed {}",
            scheduled_max_control_gap_seconds,
            plan.maximum_control_gap_seconds
        )]);
    }

    let controls: Vec<ControlDecision> = plan
        .controls
        .iter()
        .map(|control| {
            let disposition = match control.value {
                None => ControlDisposition::Missing,
                Some(value) if !value.is_finite() => ControlDisposition::Invalid,
                Some(value) if value < lower || value > upper => ControlDisposition::OutOfBand,
                Some(_) => ControlDisposition::InBand,
            };
            ControlDecision {
                control_id: control.id.clone(),
                time_seconds: control.time_seconds,
                measured_value: control.value,
                disposition,
            }
        })
        .collect();

    let mut observations = Vec::with_capacity(plan.dut_observations.len());
    for observation in &plan.dut_observations {
        let mut responsible = Vec::new();
        for (index, decision) in controls.iter().enumerate() {
            if decision.disposition.is_acceptable() {
                continue;
            }
            let previous_good = controls[..index]
                .iter()
                .rev()
                .find(|control| control.disposition.is_acceptable())
                .map_or(plan.run_start_seconds, |control| control.time_seconds);
            let next_good = controls[index + 1..]
                .iter()
                .find(|control| control.disposition.is_acceptable())
                .map_or(plan.run_end_seconds, |control| control.time_seconds);

            if observation.time_seconds >= previous_good && observation.time_seconds <= next_good {
                responsible.push(decision.control_id.clone());
            }
        }

        let disposition = if responsible.is_empty() {
            ObservationDisposition::NoOutOfBandControlDetected
        } else {
            ObservationDisposition::QuarantinedByControlExcursion
        };
        observations.push(ObservationDecision {
            observation_id: observation.id.clone(),
            time_seconds: observation.time_seconds,
            disposition,
            responsible_control_ids: responsible,
        });
    }

    Ok(DriftScreeningReport {
        guard_lower: lower,
        guard_upper: upper,
        maximum_control_gap_seconds: plan.maximum_control_gap_seconds,
        scheduled_max_control_gap_seconds,
        controls,
        observations,
        qualification: "SYNTHETIC_OR_CONTROL_SCREENING_ONLY_NOT_PHYSICAL_QUALIFICATION",
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftPlanValidationReport {
    pub plan_id: String,
    pub measurand: String,
    pub unit: String,
    pub plan_sha256: String,
    pub schema_sha256: String,
    pub screening: DriftScreeningReport,
}

impl DriftPlanValidationReport {
    /// Stable machine-readable report explicitly bounded to synthetic screening.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "verifier": "open-display-platform-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "plan_id": self.plan_id,
            "measurand": self.measurand,
            "unit": self.unit,
            "plan_sha256": self.plan_sha256,
            "schema_sha256": self.schema_sha256,
            "guard_band": {
                "lower": self.screening.guard_lower,
                "upper": self.screening.guard_upper,
                "boundary_rule": "INCLUSIVE"
            },
            "control_schedule": {
                "maximum_allowed_gap_seconds": self.screening.maximum_control_gap_seconds,
                "scheduled_max_gap_seconds": self.screening.scheduled_max_control_gap_seconds,
                "gap_scope": "run_start_to_first_control_between_control_slots_and_last_control_to_run_end",
                "claim_limit": "Planned spacing only; does not establish drift-detection power or physical stability."
            },
            "controls": self.screening.controls.iter().map(|control| serde_json::json!({
                "control_id": control.control_id,
                "time_seconds": control.time_seconds,
                "measured_value": control.measured_value,
                "disposition": control.disposition.as_str()
            })).collect::<Vec<_>>(),
            "observations": self.screening.observations.iter().map(|observation| serde_json::json!({
                "observation_id": observation.observation_id,
                "time_seconds": observation.time_seconds,
                "disposition": observation.disposition.as_str(),
                "responsible_control_ids": observation.responsible_control_ids
            })).collect::<Vec<_>>(),
            "qualification": self.screening.qualification,
            "claim_ceiling": "Synthetic control-screening algorithm only. No physical instrument, calibration, drift-detection capability or display result is qualified."
        })
    }
}

/// Validate and execute a synthetic-only drift-control plan using the checked-in schema.
///
/// This interface intentionally refuses plans labelled as physical runs. Real
/// run-record integration must be introduced through a later explicit schema
/// version after a control artifact, measurement basis, cadence and uncertainty
/// budget have been selected and reviewed.
pub fn validate_drift_control_plan_str(
    plan_text: &str,
    schema_text: &str,
) -> Result<DriftPlanValidationReport, Vec<String>> {
    let schema: serde_json::Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("drift schema JSON parse error: {error}")]),
    };
    let plan_value: serde_json::Value = match serde_json::from_str(plan_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("drift plan JSON parse error: {error}")]),
    };

    let mut errors = Vec::new();
    if schema.get("$schema").and_then(serde_json::Value::as_str)
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        errors.push("drift schema must declare JSON Schema Draft 2020-12".to_owned());
    }
    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&plan_value) {
                errors.push(format!("drift plan JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("drift schema compilation failed: {error}")),
    }

    if plan_value
        .get("record_state")
        .and_then(serde_json::Value::as_str)
        != Some("SYNTHETIC_TEST_ONLY")
    {
        errors.push("drift screening v0.1 only accepts SYNTHETIC_TEST_ONLY plans".to_owned());
    }
    if plan_value
        .get("claim_ceiling")
        .and_then(serde_json::Value::as_str)
        != Some("SYNTHETIC_OR_CONTROL_SCREENING_ONLY_NOT_PHYSICAL_QUALIFICATION")
    {
        errors.push("drift plan claim ceiling is invalid".to_owned());
    }

    let Some(controls_value) = plan_value
        .get("controls")
        .and_then(serde_json::Value::as_array)
    else {
        errors.push("drift plan controls must be an array".to_owned());
        return Err(errors);
    };
    let Some(observations_value) = plan_value
        .get("observations")
        .and_then(serde_json::Value::as_array)
    else {
        errors.push("drift plan observations must be an array".to_owned());
        return Err(errors);
    };

    let controls = controls_value
        .iter()
        .map(|control| ScheduledControl {
            id: control
                .get("control_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned(),
            time_seconds: control
                .get("time_seconds")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN),
            value: control.get("value").and_then(|value| {
                if value.is_null() {
                    None
                } else {
                    value.as_f64()
                }
            }),
        })
        .collect();
    let observations = observations_value
        .iter()
        .map(|observation| DutObservation {
            id: observation
                .get("observation_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_owned(),
            time_seconds: observation
                .get("time_seconds")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::NAN),
        })
        .collect();
    let screening = DriftControlPlan {
        run_start_seconds: plan_value
            .get("run_start_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN),
        run_end_seconds: plan_value
            .get("run_end_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN),
        nominal_value: plan_value
            .get("nominal_value")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN),
        guard_half_width: plan_value
            .get("guard_half_width")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN),
        maximum_control_gap_seconds: plan_value
            .get("maximum_control_gap_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN),
        controls,
        dut_observations: observations,
    };

    let screening = match screen_drift_controls(&screening) {
        Ok(report) => Some(report),
        Err(screening_errors) => {
            errors.extend(screening_errors);
            None
        }
    };
    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    let plan_id = plan_value
        .get("plan_id")
        .and_then(serde_json::Value::as_str)
        .expect("schema validates plan_id")
        .to_owned();
    let measurand = plan_value
        .get("measurand")
        .and_then(serde_json::Value::as_str)
        .expect("schema validates measurand")
        .to_owned();
    let unit = plan_value
        .get("unit")
        .and_then(serde_json::Value::as_str)
        .expect("schema validates unit")
        .to_owned();

    Ok(DriftPlanValidationReport {
        plan_id,
        measurand,
        unit,
        plan_sha256: super::sha256_hex(plan_text.as_bytes()),
        schema_sha256: super::sha256_hex(schema_text.as_bytes()),
        screening: screening.expect("a validated screen report must exist"),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ControlDisposition, DriftControlPlan, DutObservation, ObservationDisposition,
        ScheduledControl, screen_drift_controls, validate_drift_control_plan_str,
    };

    const DRIFT_PLAN_TEXT: &str =
        include_str!("../../../docs/engineering/display-drift-control-plan-v0.1.json");
    const DRIFT_SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/display-drift-control-plan.schema.json");

    fn control(id: &str, time_seconds: f64, value: Option<f64>) -> ScheduledControl {
        ScheduledControl {
            id: id.to_owned(),
            time_seconds,
            value,
        }
    }

    fn sample(id: &str, time_seconds: f64) -> DutObservation {
        DutObservation {
            id: id.to_owned(),
            time_seconds,
        }
    }

    fn plan(
        controls: Vec<ScheduledControl>,
        dut_observations: Vec<DutObservation>,
    ) -> DriftControlPlan {
        DriftControlPlan {
            run_start_seconds: 0.0,
            run_end_seconds: 100.0,
            nominal_value: 10.0,
            guard_half_width: 1.0,
            maximum_control_gap_seconds: 100.0,
            controls,
            dut_observations,
        }
    }

    #[test]
    fn checked_in_synthetic_plan_produces_expected_quarantine_report() {
        let report = validate_drift_control_plan_str(DRIFT_PLAN_TEXT, DRIFT_SCHEMA_TEXT)
            .expect("the checked-in synthetic plan must validate");
        assert_eq!(report.plan_id, "ODRH-DRIFT-PLAN-SYNTHETIC-001");
        assert_eq!(report.measurand, "synthetic check-standard response");
        assert_eq!(report.unit, "arbitrary_units");
        assert_eq!(report.screening.controls.len(), 3);
        assert_eq!(report.screening.maximum_control_gap_seconds, 40.0);
        assert_eq!(report.screening.scheduled_max_control_gap_seconds, 40.0);
        let report_json = report.to_json();
        assert_eq!(
            report_json["control_schedule"]["maximum_allowed_gap_seconds"].as_f64(),
            Some(40.0)
        );
        assert_eq!(
            report_json["control_schedule"]["scheduled_max_gap_seconds"].as_f64(),
            Some(40.0)
        );
        assert_eq!(
            report.screening.observations[0].disposition,
            ObservationDisposition::NoOutOfBandControlDetected
        );
        assert_eq!(
            report.screening.observations[1].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
        assert_eq!(
            report.screening.observations[1].responsible_control_ids,
            ["CONTROL-EXCURSION"]
        );
        assert_eq!(
            report.screening.observations[2].disposition,
            ObservationDisposition::NoOutOfBandControlDetected
        );
        assert!(
            report.to_json()["claim_ceiling"]
                .as_str()
                .is_some_and(|text| text.contains("No physical instrument"))
        );
    }

    #[test]
    fn drift_plan_rejects_maximum_gap_smaller_than_scheduled_gap() {
        let mut plan: serde_json::Value =
            serde_json::from_str(DRIFT_PLAN_TEXT).expect("plan fixture must parse");
        plan["maximum_control_gap_seconds"] = serde_json::json!(30.0);
        let mutated = serde_json::to_string(&plan).expect("mutated plan must serialize");
        let errors = validate_drift_control_plan_str(&mutated, DRIFT_SCHEMA_TEXT)
            .expect_err("the 40-second planned gap must not pass a 30-second cap");
        assert!(errors
            .iter()
            .any(|error| error.contains("scheduled control gap exceeds")));
    }

    #[test]
    fn drift_plan_refuses_promotion_to_a_physical_run() {
        let mut plan: serde_json::Value =
            serde_json::from_str(DRIFT_PLAN_TEXT).expect("plan fixture must parse");
        plan["record_state"] = serde_json::Value::String("ACQUIRED_UNQUALIFIED".to_owned());
        let mutated = serde_json::to_string(&plan).expect("mutated plan must serialize");
        let errors = validate_drift_control_plan_str(&mutated, DRIFT_SCHEMA_TEXT)
            .expect_err("synthetic drift algorithm must not qualify physical runs");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("SYNTHETIC_TEST_ONLY"))
        );
    }

    #[test]
    fn guard_band_boundaries_are_inclusive() {
        let report = screen_drift_controls(&plan(
            vec![
                control("lower", 10.0, Some(9.0)),
                control("upper", 90.0, Some(11.0)),
            ],
            vec![sample("inside", 50.0)],
        ))
        .expect("boundary values are valid controls");
        assert_eq!(report.guard_lower, 9.0);
        assert_eq!(report.guard_upper, 11.0);
        assert!(
            report
                .controls
                .iter()
                .all(|c| c.disposition == ControlDisposition::InBand)
        );
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::NoOutOfBandControlDetected
        );
    }

    #[test]
    fn out_of_band_control_quarantines_the_interval_between_good_anchors() {
        let report = screen_drift_controls(&plan(
            vec![
                control("pre", 10.0, Some(10.0)),
                control("excursion", 50.0, Some(12.0)),
                control("post", 80.0, Some(10.0)),
            ],
            vec![
                sample("before", 5.0),
                sample("affected", 60.0),
                sample("after", 90.0),
            ],
        ))
        .expect("control schedule should validate");
        assert_eq!(
            report.controls[1].disposition,
            ControlDisposition::OutOfBand
        );
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::NoOutOfBandControlDetected
        );
        assert_eq!(
            report.observations[1].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
        assert_eq!(report.observations[1].responsible_control_ids.len(), 1);
        assert_eq!(
            report.observations[1].responsible_control_ids[0],
            "excursion"
        );
        assert_eq!(
            report.observations[2].disposition,
            ObservationDisposition::NoOutOfBandControlDetected
        );
    }

    #[test]
    fn missing_control_is_not_interpreted_as_in_band() {
        let report = screen_drift_controls(&plan(
            vec![
                control("pre", 10.0, Some(10.0)),
                control("missed", 50.0, None),
                control("post", 80.0, Some(10.0)),
            ],
            vec![sample("affected", 50.0)],
        ))
        .expect("missing readings are represented as slots");
        assert_eq!(report.controls[1].disposition, ControlDisposition::Missing);
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
    }

    #[test]
    fn non_finite_control_value_is_invalid_and_quarantines() {
        let report = screen_drift_controls(&plan(
            vec![
                control("pre", 10.0, Some(10.0)),
                control("invalid", 50.0, Some(f64::NAN)),
                control("post", 80.0, Some(10.0)),
            ],
            vec![sample("affected", 55.0)],
        ))
        .expect("non-finite reading is preserved as invalid control evidence");
        assert_eq!(report.controls[1].disposition, ControlDisposition::Invalid);
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
    }

    #[test]
    fn first_and_last_bad_controls_extend_to_run_edges() {
        let report = screen_drift_controls(&plan(
            vec![
                control("first_bad", 10.0, Some(12.0)),
                control("good", 50.0, Some(10.0)),
                control("last_bad", 90.0, Some(8.0)),
            ],
            vec![
                sample("early", 2.0),
                sample("middle", 60.0),
                sample("late", 99.0),
            ],
        ))
        .expect("edge control excursions should validate");
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
        assert_eq!(
            report.observations[1].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
        assert_eq!(
            report.observations[2].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
    }

    #[test]
    fn duplicate_ids_unordered_controls_and_out_of_run_times_reject() {
        let duplicate = screen_drift_controls(&plan(
            vec![
                control("x", 10.0, Some(10.0)),
                control("x", 20.0, Some(10.0)),
            ],
            vec![],
        ));
        assert!(
            duplicate
                .expect_err("duplicate IDs must fail")
                .iter()
                .any(|e| e.contains("duplicate control"))
        );

        let unordered = screen_drift_controls(&plan(
            vec![
                control("later", 20.0, Some(10.0)),
                control("earlier", 10.0, Some(10.0)),
            ],
            vec![],
        ));
        assert!(
            unordered
                .expect_err("out-of-order controls must fail")
                .iter()
                .any(|e| e.contains("time-ordered"))
        );

        let outside =
            screen_drift_controls(&plan(vec![control("outside", 110.0, Some(10.0))], vec![]));
        assert!(
            outside
                .expect_err("out-of-run control must fail")
                .iter()
                .any(|e| e.contains("outside the run"))
        );
    }

    #[test]
    fn immediately_outside_both_guard_boundaries_is_out_of_band() {
        let delta = f64::EPSILON * 16.0;
        let report = screen_drift_controls(&plan(
            vec![
                control("lower_inside", 10.0, Some(9.0)),
                control("lower_outside", 20.0, Some(9.0 - delta)),
                control("upper_inside", 30.0, Some(11.0)),
                control("upper_outside", 40.0, Some(11.0 + delta)),
            ],
            vec![],
        ))
        .expect("finite values around both guard boundaries should validate");
        assert_eq!(report.controls[0].disposition, ControlDisposition::InBand);
        assert_eq!(
            report.controls[1].disposition,
            ControlDisposition::OutOfBand
        );
        assert_eq!(report.controls[2].disposition, ControlDisposition::InBand);
        assert_eq!(
            report.controls[3].disposition,
            ControlDisposition::OutOfBand
        );
    }

    #[test]
    fn adjacent_excursions_preserve_each_control_identity() {
        let report = screen_drift_controls(&plan(
            vec![
                control("pre", 10.0, Some(10.0)),
                control("bad_a", 30.0, Some(12.0)),
                control("bad_b", 50.0, None),
                control("post", 80.0, Some(10.0)),
            ],
            vec![sample("during", 40.0)],
        ))
        .expect("adjacent bad controls should be preserved");
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::QuarantinedByControlExcursion
        );
        assert_eq!(report.observations[0].responsible_control_ids.len(), 2);
        assert_eq!(report.observations[0].responsible_control_ids[0], "bad_a");
        assert_eq!(report.observations[0].responsible_control_ids[1], "bad_b");
    }

    #[test]
    fn planned_control_gap_includes_run_edges_and_rejects_sparse_schedule() {
        let mut sparse = plan(
            vec![
                control("first", 10.0, Some(10.0)),
                control("last", 90.0, Some(10.0)),
            ],
            vec![sample("middle", 50.0)],
        );
        sparse.maximum_control_gap_seconds = 30.0;
        let errors = screen_drift_controls(&sparse)
            .expect_err("a plan with an 80-second internal gap must fail a 30-second cap");
        assert!(errors.iter().any(|e| e.contains("scheduled control gap exceeds")));

        let mut edge_gap = plan(
            vec![
                control("first", 20.0, Some(10.0)),
                control("last", 90.0, Some(10.0)),
            ],
            vec![],
        );
        edge_gap.maximum_control_gap_seconds = 25.0;
        let errors = screen_drift_controls(&edge_gap)
            .expect_err("the 30-second terminal edge gap must be checked too");
        assert!(errors.iter().any(|e| e.contains("scheduled control gap exceeds")));
    }

    #[test]
    fn invalid_control_gap_bounds_are_rejected() {
        let mut invalid = plan(vec![control("only", 50.0, Some(10.0))], vec![]);
        invalid.maximum_control_gap_seconds = 0.0;
        assert!(screen_drift_controls(&invalid)
            .expect_err("zero control gap cap must fail")
            .iter()
            .any(|e| e.contains("maximum control gap must be finite and positive")));

        invalid.maximum_control_gap_seconds = f64::INFINITY;
        assert!(screen_drift_controls(&invalid)
            .expect_err("infinite control gap cap must fail")
            .iter()
            .any(|e| e.contains("maximum control gap must be finite and positive")));
    }

    #[test]
    fn no_detected_excursion_is_not_labeled_physical_pass() {
        let report = screen_drift_controls(&plan(
            vec![
                control("a", 10.0, Some(10.0)),
                control("b", 90.0, Some(10.0)),
            ],
            vec![sample("d", 50.0)],
        ))
        .expect("screening should be valid");
        assert_eq!(
            report.qualification,
            "SYNTHETIC_OR_CONTROL_SCREENING_ONLY_NOT_PHYSICAL_QUALIFICATION"
        );
        assert_eq!(
            report.observations[0].disposition,
            ObservationDisposition::NoOutOfBandControlDetected
        );
    }
}
