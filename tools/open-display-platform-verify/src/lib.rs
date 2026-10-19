#![deny(unsafe_code)]
// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed validation of the ODRH-001 architecture/parts manifest.
//!
//! A valid manifest is a valid specification contract only. It does not prove
//! that hardware exists, can be built, is optically calibrated, or is qualified.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const EXPECTED_MANIFEST_ID: &str = "ODRH-001";
pub const EXPECTED_SCHEMA_VERSION: &str = "0.1.0";
pub const EXPECTED_SCHEMA_DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";

mod cad_receipt;
pub use cad_receipt::{CadRuntimeReceiptValidationReport, validate_cad_runtime_receipt_str};
pub mod drift_control;
pub use drift_control::{DriftPlanValidationReport, validate_drift_control_plan_str};
pub mod energy;
pub mod geometry;
pub use geometry::{
    evaluate_geometry_screen, validate_geometry_screen_str, GeometryScreenInput,
    GeometryScreenResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub manifest_id: String,
    pub schema_version: String,
    pub contract_state: String,
    pub module_count: usize,
    pub measurement_channel_count: usize,
    pub benchmark_count: usize,
    pub gate_count: usize,
    pub reference_count: usize,
    pub manifest_sha256: String,
    pub schema_sha256: String,
}

impl ValidationReport {
    /// Stable machine-readable output with an explicit non-qualification ceiling.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verifier": "open-display-platform-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "schema_draft": EXPECTED_SCHEMA_DRAFT,
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "contract_state": self.contract_state,
            "module_count": self.module_count,
            "measurement_channel_count": self.measurement_channel_count,
            "benchmark_count": self.benchmark_count,
            "gate_count": self.gate_count,
            "reference_count": self.reference_count,
            "manifest_sha256": self.manifest_sha256,
            "schema_sha256": self.schema_sha256,
            "physical_validation": "NOT_PERFORMED",
            "claim_ceiling": "Specification contract only. No physical build, optical measurement, safety review, supply-chain validation, or qualification is implied."
        })
    }
}

/// Validate the supplied manifest against Draft 2020-12 and independent semantic gates.
///
/// # Errors
///
/// Returns an ordered, deduplicated list of schema, identity, provenance, or
/// fail-closed status violations.
pub fn validate_platform_str(
    manifest_text: &str,
    schema_text: &str,
) -> Result<ValidationReport, Vec<String>> {
    let schema: Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("schema JSON parse error: {error}")]),
    };
    let manifest: Value = match serde_json::from_str(manifest_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("manifest JSON parse error: {error}")]),
    };

    let mut errors = Vec::new();
    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_DRAFT) {
        errors.push(format!("schema must declare {EXPECTED_SCHEMA_DRAFT}"));
    }

    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&manifest) {
                errors.push(format!("JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("schema compilation failed: {error}")),
    }

    errors.extend(validate_semantics(&manifest));
    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(ValidationReport {
        manifest_id: EXPECTED_MANIFEST_ID.to_owned(),
        schema_version: EXPECTED_SCHEMA_VERSION.to_owned(),
        contract_state: "CONTRACT_VALID_UNQUALIFIED".to_owned(),
        module_count: count_array(&manifest, "modules"),
        measurement_channel_count: count_array(&manifest, "measurement_channels"),
        benchmark_count: count_array(&manifest, "benchmarks"),
        gate_count: count_array(&manifest, "gates"),
        reference_count: count_array(&manifest, "references"),
        manifest_sha256: sha256_hex(manifest_text.as_bytes()),
        schema_sha256: sha256_hex(schema_text.as_bytes()),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasurementRunValidationReport {
    pub record_id: String,
    pub record_state: String,
    pub observation_count: usize,
    pub instrument_count: usize,
    pub raw_artifact_count: usize,
    pub manifest_sha256: String,
    pub schema_sha256: String,
}

impl MeasurementRunValidationReport {
    /// Stable JSON report. Contract validity does not imply a physical result passed.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verifier": "open-display-platform-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "record_id": self.record_id,
            "record_state": self.record_state,
            "contract_state": "RUN_CONTRACT_VALID_UNQUALIFIED",
            "observation_count": self.observation_count,
            "instrument_count": self.instrument_count,
            "raw_artifact_count": self.raw_artifact_count,
            "manifest_sha256": self.manifest_sha256,
            "schema_sha256": self.schema_sha256,
            "physical_validation": "NOT_PERFORMED_OR_REPORTED_SEPARATELY",
            "claim_ceiling": "The run record conforms to its schema/semantic contract; physical accuracy, safety and qualification require separate evidence."
        })
    }
}

/// Validate a run record and apply the cross-reference, calibration, uncertainty,
/// and status-promotion checks that cannot be expressed safely by schema alone.
///
/// # Errors
///
/// Returns a sorted, deduplicated set of schema and semantic violations.
pub fn validate_measurement_run_str(
    record_text: &str,
    schema_text: &str,
) -> Result<MeasurementRunValidationReport, Vec<String>> {
    let schema: Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("run schema JSON parse error: {error}")]),
    };
    let record: Value = match serde_json::from_str(record_text) {
        Ok(value) => value,
        Err(error) => return Err(vec![format!("run record JSON parse error: {error}")]),
    };

    let mut errors = Vec::new();
    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_DRAFT) {
        errors.push(format!("run schema must declare {EXPECTED_SCHEMA_DRAFT}"));
    }
    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&record) {
                errors.push(format!("run JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("run schema compilation failed: {error}")),
    }
    errors.extend(validate_measurement_run_semantics(&record));
    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(MeasurementRunValidationReport {
        record_id: record
            .get("record_id")
            .and_then(Value::as_str)
            .expect("schema validates record_id")
            .to_owned(),
        record_state: record
            .get("record_state")
            .and_then(Value::as_str)
            .expect("schema validates record_state")
            .to_owned(),
        observation_count: count_array(&record, "observations"),
        instrument_count: count_array(&record, "instruments"),
        raw_artifact_count: count_array(&record, "raw_artifacts"),
        manifest_sha256: sha256_hex(record_text.as_bytes()),
        schema_sha256: sha256_hex(schema_text.as_bytes()),
    })
}

fn approximately_equal(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1e-12_f64.max(actual.abs().max(expected.abs()) * 1e-9)
}

/// Checks nested uncertainty-budget provenance and, for a linearized budget,
/// independently recomputes propagated uncertainty from standard components
/// and covariance contributions already projected into the measurand's units.
fn validate_uncertainty_budget(
    observation: &Value,
    observation_id: &str,
    observation_status: &str,
) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(budget) = observation.get("uncertainty_budget") else {
        return vec![format!(
            "observation {observation_id} has no uncertainty budget record"
        )];
    };

    let budget_status = budget
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("INVALID");
    let method = budget
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("INVALID");
    let terms = budget
        .get("terms")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let covariance_terms = budget
        .get("covariance_terms")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    validate_unique_field(terms, "uncertainty component", "term_id", &mut errors);

    let term_ids: BTreeSet<String> = terms
        .iter()
        .filter_map(|term| term.get("term_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let mut covariance_pairs = BTreeSet::new();
    for covariance in covariance_terms {
        let first = covariance
            .get("term_a_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let second = covariance
            .get("term_b_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if first == second {
            errors.push(format!(
                "uncertainty budget for {observation_id} cannot declare self-covariance for {first}"
            ));
        }
        for component_id in [first, second] {
            if !term_ids.contains(component_id) {
                errors.push(format!(
                    "uncertainty budget for {observation_id} references unknown component {component_id}"
                ));
            }
        }
        let pair = if first <= second {
            (first.to_owned(), second.to_owned())
        } else {
            (second.to_owned(), first.to_owned())
        };
        if !covariance_pairs.insert(pair) {
            errors.push(format!(
                "uncertainty budget for {observation_id} repeats a covariance pair"
            ));
        }
    }

    if observation_status == "NOT_EVALUATED" {
        if budget_status != "NOT_EVALUATED"
            || method != "NOT_EVALUATED"
            || !terms.is_empty()
            || !covariance_terms.is_empty()
            || !budget
                .get("combined_standard_uncertainty")
                .is_some_and(Value::is_null)
            || !budget.get("coverage_factor").is_some_and(Value::is_null)
            || !budget
                .get("expanded_uncertainty")
                .is_some_and(Value::is_null)
        {
            errors.push(format!(
                "NOT_EVALUATED observation {observation_id} must carry an empty, NOT_EVALUATED uncertainty budget"
            ));
        }
        return errors;
    }

    let expected_budget_status = match observation_status {
        "MEASURED_UNQUALIFIED" => "CALCULATED_UNREVIEWED",
        "INDEPENDENTLY_REVIEWED" => "INDEPENDENTLY_REVIEWED",
        _ => "INVALID",
    };
    if budget_status != expected_budget_status {
        errors.push(format!(
            "uncertainty budget for {observation_id} has status {budget_status} inconsistent with observation status {observation_status}"
        ));
    }
    if method == "NOT_EVALUATED" {
        errors.push(format!(
            "measured observation {observation_id} cannot use a NOT_EVALUATED uncertainty method"
        ));
    }
    if terms.is_empty() {
        errors.push(format!(
            "measured observation {observation_id} needs at least one uncertainty component"
        ));
    }

    let budget_expanded = budget.get("expanded_uncertainty").and_then(Value::as_f64);
    let top_level_expanded = observation
        .get("expanded_uncertainty")
        .and_then(Value::as_f64);
    if budget_expanded.is_none_or(|value| !value.is_finite() || value < 0.0) {
        errors.push(format!(
            "uncertainty budget for {observation_id} needs a finite nonnegative expanded uncertainty"
        ));
    }
    if let (Some(budget_value), Some(top_level_value)) = (budget_expanded, top_level_expanded)
        && !approximately_equal(budget_value, top_level_value)
    {
        errors.push(format!(
            "uncertainty budget expanded uncertainty disagrees with observation {observation_id}"
        ));
    }

    if method == "LINEARIZED_COVARIANCE" {
        let combined = budget
            .get("combined_standard_uncertainty")
            .and_then(Value::as_f64);
        let coverage = budget.get("coverage_factor").and_then(Value::as_f64);
        if combined.is_none_or(|value| !value.is_finite() || value < 0.0) {
            errors.push(format!(
                "linearized uncertainty budget for {observation_id} needs finite nonnegative combined standard uncertainty"
            ));
        }
        if coverage.is_none_or(|value| !value.is_finite() || value <= 0.0) {
            errors.push(format!(
                "linearized uncertainty budget for {observation_id} needs a finite positive coverage factor"
            ));
        }

        let mut variance = 0.0_f64;
        for term in terms {
            let term_id = term
                .get("term_id")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            let standard = term.get("standard_uncertainty").and_then(Value::as_f64);
            let sensitivity = term.get("sensitivity_coefficient").and_then(Value::as_f64);
            match (standard, sensitivity) {
                (Some(u), Some(coefficient))
                    if u.is_finite() && u >= 0.0 && coefficient.is_finite() =>
                {
                    let propagated = u * coefficient;
                    variance += propagated * propagated;
                    if !variance.is_finite() {
                        errors.push(format!(
                            "uncertainty variance overflow for component {term_id} in {observation_id}"
                        ));
                    }
                }
                _ => errors.push(format!(
                    "uncertainty component {term_id} has invalid standard uncertainty or sensitivity coefficient"
                )),
            }
        }
        for covariance in covariance_terms {
            if let Some(value) = covariance
                .get("covariance_contribution")
                .and_then(Value::as_f64)
            {
                variance += 2.0 * value;
            } else {
                errors.push(format!(
                    "uncertainty covariance contribution for {observation_id} is not numeric"
                ));
            }
        }

        if !variance.is_finite() || variance < 0.0 {
            errors.push(format!(
                "linearized uncertainty budget for {observation_id} has negative or non-finite total variance"
            ));
        } else {
            let expected_combined = variance.sqrt();
            if let Some(reported_combined) = combined
                && !approximately_equal(reported_combined, expected_combined)
            {
                errors.push(format!(
                    "reported combined standard uncertainty for {observation_id} disagrees with recomputed covariance budget"
                ));
            }
            if let Some(coverage_factor) = coverage {
                let expected_expanded = coverage_factor * expected_combined;
                if let Some(reported_expanded) = budget_expanded
                    && !approximately_equal(reported_expanded, expected_expanded)
                {
                    errors.push(format!(
                        "reported expanded uncertainty for {observation_id} disagrees with coverage factor times combined uncertainty"
                    ));
                }
            }
        }
    } else if !covariance_terms.is_empty() {
        errors.push(format!(
            "uncertainty method {method} for {observation_id} cannot attach linearized covariance terms"
        ));
    }

    errors
}

/// Parse a fixed-width ASCII decimal field without accepting signs or whitespace.
fn parse_ascii_decimal(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    bytes.iter().try_fold(0_u32, |value, byte| {
        value.checked_mul(10)?.checked_add(u32::from(*byte - b'0'))
    })
}

/// Validate a Gregorian YYYY-MM-DD calendar date.
fn is_valid_iso_date_bytes(bytes: &[u8]) -> bool {
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let (Some(year), Some(month), Some(day)) = (
        parse_ascii_decimal(&bytes[0..4]),
        parse_ascii_decimal(&bytes[5..7]),
        parse_ascii_decimal(&bytes[8..10]),
    ) else {
        return false;
    };
    if !(1..=12).contains(&month) {
        return false;
    }
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

fn is_valid_iso_date(value: &str) -> bool {
    is_valid_iso_date_bytes(value.as_bytes())
}

/// Validate the RFC 3339 date-time syntax used by run timestamps.
///
/// JSON Schema Draft 2020-12 treats format as annotation-only by default.
/// This application-level check validates calendar dates, time fields,
/// fractional seconds, and an explicit timezone rather than trusting annotation.
fn is_valid_rfc3339_datetime(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || !is_valid_iso_date_bytes(&bytes[..10])
        || !matches!(bytes[10], b'T' | b't')
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return false;
    }

    let (Some(hour), Some(minute), Some(second)) = (
        parse_ascii_decimal(&bytes[11..13]),
        parse_ascii_decimal(&bytes[14..16]),
        parse_ascii_decimal(&bytes[17..19]),
    ) else {
        return false;
    };
    if hour > 23 || minute > 59 || second > 60 {
        return false;
    }

    let mut index = 19;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == fraction_start {
            return false;
        }
    }

    match bytes.get(index) {
        Some(b'Z' | b'z') => index + 1 == bytes.len(),
        Some(b'+' | b'-') => {
            if bytes.len() != index + 6 || bytes[index + 3] != b':' {
                return false;
            }
            let (Some(offset_hour), Some(offset_minute)) = (
                parse_ascii_decimal(&bytes[index + 1..index + 3]),
                parse_ascii_decimal(&bytes[index + 4..index + 6]),
            ) else {
                return false;
            };
            offset_hour <= 23 && offset_minute <= 59
        }
        _ => false,
    }
}

fn validate_measurement_run_semantics(record: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let record_state = record
        .get("record_state")
        .and_then(Value::as_str)
        .unwrap_or("INVALID");
    let physical_state = record
        .get("physical_validation")
        .and_then(Value::as_str)
        .unwrap_or("INVALID");
    let acquired_at = record.get("acquired_at").and_then(Value::as_str);
    let run_date = acquired_at.and_then(|timestamp| timestamp.get(..10));
    if acquired_at.is_some_and(|timestamp| !is_valid_rfc3339_datetime(timestamp)) {
        errors.push(
            "acquired_at must be a valid RFC 3339 date-time with an explicit timezone".to_owned(),
        );
    }
    if record
        .get("review")
        .and_then(Value::as_object)
        .and_then(|review| review.get("reviewed_at"))
        .and_then(Value::as_str)
        .is_some_and(|timestamp| !is_valid_rfc3339_datetime(timestamp))
    {
        errors.push(
            "review.reviewed_at must be a valid RFC 3339 date-time with an explicit timezone"
                .to_owned(),
        );
    }

    let instruments = record
        .get("instruments")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let observations = record
        .get("observations")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let artifacts = record
        .get("raw_artifacts")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let exclusions = record
        .get("exclusions")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    validate_unique_field(instruments, "run instrument", "instrument_id", &mut errors);
    validate_unique_field(
        observations,
        "run observation",
        "observation_id",
        &mut errors,
    );
    validate_unique_field(artifacts, "raw artifact", "artifact_id", &mut errors);
    validate_unique_field(
        exclusions,
        "observation exclusion",
        "exclusion_id",
        &mut errors,
    );

    let instrument_ids: BTreeSet<String> = instruments
        .iter()
        .filter_map(|item| item.get("instrument_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let artifact_ids: BTreeSet<String> = artifacts
        .iter()
        .filter_map(|item| item.get("artifact_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let observation_ids: BTreeSet<String> = observations
        .iter()
        .filter_map(|item| item.get("observation_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let mut measured_instrument_ids = BTreeSet::new();
    for observation in observations {
        if observation.get("status").and_then(Value::as_str) != Some("NOT_EVALUATED")
            && let Some(ids) = observation.get("instrument_ids").and_then(Value::as_array)
        {
            measured_instrument_ids.extend(ids.iter().filter_map(Value::as_str).map(str::to_owned));
        }
    }

    for instrument in instruments {
        let instrument_id = instrument
            .get("instrument_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if instrument
            .get("calibration_valid_through")
            .and_then(Value::as_str)
            .is_some_and(|date| !is_valid_iso_date(date))
        {
            errors.push(format!(
                "instrument {instrument_id} has an invalid calibration_valid_through date"
            ));
        }
        if instrument.get("calibration_state").and_then(Value::as_str) == Some("VALID")
            && (instrument
                .get("calibration_record_digest")
                .and_then(Value::as_str)
                .is_none()
                || instrument
                    .get("calibration_valid_through")
                    .and_then(Value::as_str)
                    .is_none())
        {
            errors.push(format!(
                "instrument {instrument_id} cannot claim VALID calibration without record digest and expiry"
            ));
        }

        if measured_instrument_ids.contains(instrument_id) {
            let calibration_digest = instrument
                .get("calibration_record_digest")
                .and_then(Value::as_str);
            let calibration_artifact_id = instrument
                .get("calibration_record_artifact_id")
                .and_then(Value::as_str);
            match (calibration_digest, calibration_artifact_id) {
                (Some(expected_digest), Some(artifact_id)) => {
                    match artifacts.iter().find(|artifact| {
                        artifact.get("artifact_id").and_then(Value::as_str) == Some(artifact_id)
                    }) {
                        Some(artifact) => {
                            if artifact.get("role").and_then(Value::as_str)
                                != Some("INSTRUMENT_CALIBRATION")
                            {
                                errors.push(format!(
                                    "instrument {instrument_id} calibration artifact {artifact_id} has the wrong role"
                                ));
                            }
                            if artifact.get("sha256").and_then(Value::as_str)
                                != Some(expected_digest)
                            {
                                errors.push(format!(
                                    "instrument {instrument_id} calibration digest does not match artifact {artifact_id}"
                                ));
                            }
                        }
                        None => errors.push(format!(
                            "instrument {instrument_id} references unresolved calibration artifact {artifact_id}"
                        )),
                    }
                }
                _ => errors.push(format!(
                    "measured instrument {instrument_id} must bind its calibration record to a raw artifact and digest"
                )),
            }

            let profile_status = instrument
                .get("capability_profile_status")
                .and_then(Value::as_str)
                .unwrap_or("UNASSESSED");
            if !matches!(
                profile_status,
                "DECLARED_UNREVIEWED" | "INDEPENDENTLY_REVIEWED"
            ) {
                errors.push(format!(
                    "measured instrument {instrument_id} has no decision-usable capability profile"
                ));
            }

            let profile_digest = instrument
                .get("capability_profile_sha256")
                .and_then(Value::as_str);
            let profile_artifact_id = instrument
                .get("capability_profile_artifact_id")
                .and_then(Value::as_str);
            match (profile_digest, profile_artifact_id) {
                (Some(expected_digest), Some(artifact_id)) => {
                    match artifacts.iter().find(|artifact| {
                        artifact.get("artifact_id").and_then(Value::as_str) == Some(artifact_id)
                    }) {
                        Some(artifact) => {
                            if artifact.get("role").and_then(Value::as_str)
                                != Some("INSTRUMENT_CAPABILITY_PROFILE")
                            {
                                errors.push(format!(
                                    "instrument {instrument_id} capability artifact {artifact_id} has the wrong role"
                                ));
                            }
                            if artifact.get("sha256").and_then(Value::as_str)
                                != Some(expected_digest)
                            {
                                errors.push(format!(
                                    "instrument {instrument_id} capability-profile digest does not match artifact {artifact_id}"
                                ));
                            }
                        }
                        None => errors.push(format!(
                            "instrument {instrument_id} references unresolved capability-profile artifact {artifact_id}"
                        )),
                    }
                }
                _ => errors.push(format!(
                    "measured instrument {instrument_id} must bind its capability profile to a raw artifact and digest"
                )),
            }
        }
    }

    let mut measured_count = 0_usize;
    let mut independently_reviewed_count = 0_usize;
    for observation in observations {
        let observation_id = observation
            .get("observation_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let measurand = observation
            .get("measurand")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let unit = observation
            .get("unit")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let status = observation
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("INVALID");
        let used_instruments: Vec<&str> = observation
            .get("instrument_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let raw_ids: Vec<&str> = observation
            .get("raw_artifact_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();

        errors.extend(validate_uncertainty_budget(
            observation,
            observation_id,
            status,
        ));

        for id in &used_instruments {
            if !instrument_ids.contains(*id) {
                errors.push(format!(
                    "observation {observation_id} references unresolved instrument {id}"
                ));
            }
        }
        for id in &raw_ids {
            if !artifact_ids.contains(*id) {
                errors.push(format!(
                    "observation {observation_id} references unresolved raw artifact {id}"
                ));
            }
        }

        if status == "NOT_EVALUATED" {
            if observation
                .get("value")
                .is_some_and(|value| !value.is_null())
                || observation
                    .get("expanded_uncertainty")
                    .is_some_and(|value| !value.is_null())
                || observation
                    .get("minimum_decision_effect")
                    .is_some_and(|value| !value.is_null())
                || observation
                    .get("decision_capability")
                    .and_then(Value::as_str)
                    != Some("NOT_EVALUATED")
            {
                errors.push(format!(
                    "NOT_EVALUATED observation {observation_id} cannot claim a derived value, uncertainty, MDE, or decision capability"
                ));
            }
            continue;
        }

        measured_count += 1;
        if status == "INDEPENDENTLY_REVIEWED" {
            independently_reviewed_count += 1;
        }
        let value = observation.get("value").and_then(Value::as_f64);
        let uncertainty = observation
            .get("expanded_uncertainty")
            .and_then(Value::as_f64);
        let effect = observation
            .get("minimum_decision_effect")
            .and_then(Value::as_f64);
        if value.is_none_or(|number| !number.is_finite()) {
            errors.push(format!(
                "measured observation {observation_id} needs a finite numeric value"
            ));
        }
        if uncertainty.is_none_or(|number| !number.is_finite() || number < 0.0) {
            errors.push(format!(
                "measured observation {observation_id} needs finite nonnegative expanded uncertainty"
            ));
        }
        if effect.is_none_or(|number| !number.is_finite() || number <= 0.0) {
            errors.push(format!(
                "measured observation {observation_id} needs a finite positive minimum decision effect"
            ));
        }
        if used_instruments.is_empty() {
            errors.push(format!(
                "measured observation {observation_id} must name its instruments"
            ));
        }
        if raw_ids.is_empty() {
            errors.push(format!(
                "measured observation {observation_id} must link raw artifacts"
            ));
        }

        for id in &used_instruments {
            if let Some(instrument) = instruments.iter().find(|instrument| {
                instrument.get("instrument_id").and_then(Value::as_str) == Some(*id)
            }) {
                if instrument.get("calibration_state").and_then(Value::as_str) != Some("VALID") {
                    errors.push(format!(
                        "measured observation {observation_id} uses instrument {id} without VALID calibration state"
                    ));
                }
                if status == "INDEPENDENTLY_REVIEWED"
                    && instrument
                        .get("capability_profile_status")
                        .and_then(Value::as_str)
                        != Some("INDEPENDENTLY_REVIEWED")
                {
                    errors.push(format!(
                        "independently reviewed observation {observation_id} uses instrument {id} whose capability profile is not independently reviewed"
                    ));
                }
                if instrument
                    .get("calibration_valid_through")
                    .and_then(Value::as_str)
                    .zip(run_date)
                    .is_some_and(|(expiry, date)| expiry < date)
                {
                    errors.push(format!(
                        "measured observation {observation_id} uses instrument {id} after its calibration expiry"
                    ));
                }
                let supports_measurand_and_unit = instrument
                    .get("supported_measurands")
                    .and_then(Value::as_array)
                    .is_some_and(|measurands| {
                        measurands.iter().any(|item| {
                            item.get("measurand").and_then(Value::as_str) == Some(measurand)
                                && item.get("unit").and_then(Value::as_str) == Some(unit)
                        })
                    });
                if !supports_measurand_and_unit {
                    errors.push(format!(
                        "instrument {id} does not declare support for measurand {measurand} in unit {unit}"
                    ));
                }
            }
        }

        match (
            uncertainty,
            effect,
            observation.get("decision_capability").and_then(Value::as_str),
        ) {
            (Some(u), Some(mde), Some("CAPABLE")) if u > mde / 3.0 => errors.push(format!(
                "observation {observation_id} claims CAPABLE despite uncertainty exceeding MDE/3"
            )),
            (Some(u), Some(mde), Some("INADEQUATE")) if u <= mde / 3.0 => errors.push(format!(
                "observation {observation_id} claims INADEQUATE despite meeting uncertainty <= MDE/3"
            )),
            (_, _, Some("NOT_EVALUATED")) => errors.push(format!(
                "measured observation {observation_id} cannot leave decision capability NOT_EVALUATED"
            )),
            _ => {}
        }

        if status == "INDEPENDENTLY_REVIEWED" && record_state != "INDEPENDENTLY_REVIEWED" {
            errors.push(format!(
                "observation {observation_id} cannot be independently reviewed in record state {record_state}"
            ));
        }
    }

    for exclusion in exclusions {
        let exclusion_id = exclusion
            .get("exclusion_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let observation_id = exclusion
            .get("observation_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if !observation_ids.contains(observation_id) {
            errors.push(format!(
                "exclusion {exclusion_id} references unresolved observation {observation_id}"
            ));
        }
    }

    match record_state {
        "PLAN_ONLY" => {
            if physical_state != "NOT_PERFORMED"
                || acquired_at.is_some()
                || measured_count != 0
                || !artifacts.is_empty()
                || !exclusions.is_empty()
                || record.get("review").is_some_and(|value| !value.is_null())
            {
                errors.push(
                    "PLAN_ONLY record must have no acquisition time, measured observations, artifacts, exclusions, or review".to_owned(),
                );
            }
        }
        "ACQUIRED_UNQUALIFIED" => {
            if physical_state != "MEASURED_UNQUALIFIED"
                || acquired_at.is_none()
                || measured_count == 0
            {
                errors.push(
                    "ACQUIRED_UNQUALIFIED record needs acquisition time, measured data, and MEASURED_UNQUALIFIED physical state".to_owned(),
                );
            }
        }
        "INDEPENDENTLY_REVIEWED" => {
            if physical_state != "INDEPENDENTLY_REVIEWED"
                || acquired_at.is_none()
                || measured_count == 0
                || independently_reviewed_count != measured_count
                || record.get("review").is_none_or(Value::is_null)
            {
                errors.push(
                    "INDEPENDENTLY_REVIEWED record needs acquisition, reviewed observations, a review record, and matching physical state".to_owned(),
                );
            }
        }
        "ABORTED" => {
            if measured_count == 0 && artifacts.is_empty() && acquired_at.is_some() {
                errors.push(
                    "ABORTED record with no acquired data must not claim an acquisition timestamp"
                        .to_owned(),
                );
            }
        }
        _ => errors.push(format!("unknown run record state: {record_state}")),
    }

    errors
}

fn validate_semantics(manifest: &Value) -> Vec<String> {
    let mut errors = Vec::new();

    if manifest.get("manifest_id").and_then(Value::as_str) != Some(EXPECTED_MANIFEST_ID) {
        errors.push("manifest identity must remain ODRH-001".to_owned());
    }
    if manifest.get("schema_version").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_VERSION) {
        errors.push("v0.1 manifest schema_version must remain 0.1.0".to_owned());
    }
    if manifest.get("status").and_then(Value::as_str) != Some("SPECIFICATION_ONLY") {
        errors.push("v0.1 manifest must remain SPECIFICATION_ONLY".to_owned());
    }
    if manifest.get("physical_validation").and_then(Value::as_str) != Some("NOT_PERFORMED") {
        errors.push("physical_validation cannot be promoted by a manifest edit".to_owned());
    }

    let modules = manifest.get("modules").and_then(Value::as_array);
    let channels = manifest
        .get("measurement_channels")
        .and_then(Value::as_array);
    let benchmarks = manifest.get("benchmarks").and_then(Value::as_array);
    let gates = manifest.get("gates").and_then(Value::as_array);
    let references = manifest.get("references").and_then(Value::as_array);

    if let Some(modules) = modules {
        validate_unique_ids(modules, "module", &mut errors);
        let mut module_ids = BTreeSet::new();
        for module in modules {
            if let Some(id) = module.get("id").and_then(Value::as_str) {
                module_ids.insert(id.to_owned());
            }

            let openness = module
                .get("current_openness_class")
                .and_then(Value::as_str)
                .unwrap_or("UNRESOLVED");
            let license_status = module
                .get("license_status")
                .and_then(Value::as_str)
                .unwrap_or("UNRESOLVED");
            let has_design_source = module.get("design_source").is_some_and(|value| {
                value
                    .as_str()
                    .is_some_and(|source| !source.trim().is_empty())
            });

            if matches!(openness, "PROJECT_CONTROLLED_OPEN" | "THIRD_PARTY_OPEN")
                && (license_status != "OPEN_VERIFIED" || !has_design_source)
            {
                let id = module
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("<missing>");
                errors.push(format!(
                    "module {id} cannot claim {openness} without OPEN_VERIFIED license and design source"
                ));
            }

            if openness == "THIRD_PARTY_DOCUMENTED" {
                let has_interface_source =
                    module
                        .get("datasheet_or_interface_source")
                        .is_some_and(|value| {
                            value
                                .as_str()
                                .is_some_and(|source| !source.trim().is_empty())
                        });
                if !has_interface_source {
                    let id = module
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("<missing>");
                    errors.push(format!(
                        "module {id} cannot claim THIRD_PARTY_DOCUMENTED without a source URL"
                    ));
                }
            }

            if manifest.get("status").and_then(Value::as_str) == Some("SPECIFICATION_ONLY")
                && module.get("module_status").and_then(Value::as_str) == Some("QUALIFIED")
            {
                let id = module
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("<missing>");
                errors.push(format!(
                    "specification-only module {id} cannot be QUALIFIED"
                ));
            }

            if let Some(parameters) = module.get("design_parameters").and_then(Value::as_array) {
                let mut parameter_names = BTreeSet::new();
                for parameter in parameters {
                    let name = parameter
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("<missing>");
                    if !parameter_names.insert(name.to_owned()) {
                        errors.push(format!(
                            "module design parameter name must be unique: {name}"
                        ));
                    }
                    if manifest.get("status").and_then(Value::as_str) == Some("SPECIFICATION_ONLY")
                        && parameter.get("status").and_then(Value::as_str) != Some("EXAMPLE_ONLY")
                    {
                        errors.push(format!(
                            "specification-only design parameter {name} must remain EXAMPLE_ONLY"
                        ));
                    }
                }
            }
        }

        if let Some(channels) = channels {
            validate_unique_ids(channels, "measurement channel", &mut errors);
            for channel in channels {
                let channel_id = channel
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("<missing>");
                let source = channel
                    .get("source_module_id")
                    .and_then(Value::as_str)
                    .unwrap_or("<missing>");
                if !module_ids.contains(source) {
                    errors.push(format!(
                        "measurement channel {channel_id} has unresolved source module {source}"
                    ));
                }
            }
        } else {
            errors.push("measurement_channels must be an array".to_owned());
        }
    } else {
        errors.push("modules must be an array".to_owned());
    }

    if let Some(benchmarks) = benchmarks {
        validate_unique_ids(benchmarks, "benchmark", &mut errors);
        if manifest.get("status").and_then(Value::as_str) == Some("SPECIFICATION_ONLY") {
            for benchmark in benchmarks {
                if benchmark.get("evidence_status").and_then(Value::as_str) != Some("NOT_MEASURED")
                {
                    let id = benchmark
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("<missing>");
                    errors.push(format!(
                        "specification-only benchmark {id} must remain NOT_MEASURED"
                    ));
                }
            }
        }
    } else {
        errors.push("benchmarks must be an array".to_owned());
    }

    if let Some(gates) = gates {
        validate_unique_ids(gates, "gate", &mut errors);
        if manifest.get("status").and_then(Value::as_str) == Some("SPECIFICATION_ONLY") {
            for gate in gates {
                if gate.get("status").and_then(Value::as_str) != Some("NOT_STARTED") {
                    let id = gate
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("<missing>");
                    errors.push(format!(
                        "specification-only gate {id} must remain NOT_STARTED"
                    ));
                }
            }
        }
    } else {
        errors.push("gates must be an array".to_owned());
    }

    if let Some(references) = references {
        validate_unique_ids(references, "reference", &mut errors);
    } else {
        errors.push("references must be an array".to_owned());
    }

    if manifest
        .pointer("/license_policy/oshwa_certification")
        .and_then(Value::as_str)
        != Some("NOT_CLAIMED")
    {
        errors.push(
            "OSHWA certification must remain NOT_CLAIMED until separately reviewed".to_owned(),
        );
    }

    errors
}

fn validate_unique_ids(records: &[Value], category: &str, errors: &mut Vec<String>) {
    validate_unique_field(records, category, "id", errors);
}

fn validate_unique_field(records: &[Value], category: &str, field: &str, errors: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for record in records {
        let id = record
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if !seen.insert(id.to_owned()) {
            errors.push(format!("duplicate {category} ID: {id}"));
        }
    }
}

fn count_array(value: &Value, key: &str) -> usize {
    value.get(key).and_then(Value::as_array).map_or(0, Vec::len)
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        is_valid_iso_date, is_valid_rfc3339_datetime, sha256_hex, validate_measurement_run_str,
        validate_platform_str,
    };
    use serde_json::Value;

    const MANIFEST_TEXT: &str =
        include_str!("../../../docs/engineering/open-display-reference-platform-v0.1.json");
    const SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/open-display-reference-platform.schema.json");
    const CAD_SOURCE: &str = include_str!(
        "../../../hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad"
    );
    const CAD_CHECK_TEXT: &str = include_str!(
        "../../../hardware/open-display-reference/validation/odrh_sensor_plate_cad_check-v0.1.json"
    );
    const HARDWARE_LICENSE_TEXT: &str =
        include_str!("../../../hardware/open-display-reference/LICENSE");
    const HARDWARE_README_TEXT: &str =
        include_str!("../../../hardware/open-display-reference/README.md");
    const RUN_TEXT: &str =
        include_str!("../../../docs/engineering/display-measurement-run-template-v0.1.json");
    const RUN_SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/display-measurement-run.schema.json");

    fn parse_manifest() -> Value {
        serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest must parse")
    }

    fn encoded(value: &Value) -> String {
        serde_json::to_string(value).expect("mutated fixture must serialize")
    }

    fn parse_run() -> Value {
        serde_json::from_str(RUN_TEXT).expect("run template must parse")
    }

    #[test]
    fn cad_receipt_binds_exact_native_source_and_preserves_physical_claim_ceiling() {
        let receipt: Value =
            serde_json::from_str(CAD_CHECK_TEXT).expect("CAD validation receipt must parse");
        let source_hash = sha256_hex(CAD_SOURCE.as_bytes());

        assert!(CAD_SOURCE.contains("SPDX-License-Identifier: CERN-OHL-S-2.0"));
        assert!(
            HARDWARE_LICENSE_TEXT
                .contains("CERN Open Hardware Licence Version 2 - Strongly Reciprocal")
        );
        assert!(HARDWARE_README_TEXT.contains("[LICENSE](LICENSE)"));

        assert_eq!(
            receipt
                .pointer("/source_revision/sha256")
                .and_then(Value::as_str),
            Some(source_hash.as_str())
        );
        assert_eq!(
            receipt.pointer("/source_path").and_then(Value::as_str),
            Some("hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad")
        );
        assert_eq!(
            receipt
                .pointer("/claim_boundaries/physical_validation")
                .and_then(Value::as_str),
            Some("NOT_PERFORMED")
        );
        assert_eq!(
            receipt
                .pointer("/claim_boundaries/overall_disposition")
                .and_then(Value::as_str),
            Some("GEOMETRY_EXPORT_CHECKED_NOT_PHYSICALLY_QUALIFIED")
        );
        assert_eq!(
            receipt
                .pointer("/execution/tested_commit_sha")
                .and_then(Value::as_str),
            Some("c417ef92a60309c7f4647b524c4f00a6fd91bfd1")
        );

        assert_eq!(
            receipt
                .pointer("/claim_boundaries/preview_rendered")
                .and_then(Value::as_bool),
            Some(false),
            "the historical workflow did not render a preview"
        );
        assert!(
            receipt
                .get("correction_notes")
                .and_then(Value::as_array)
                .is_some_and(|notes| notes.iter().any(|note| {
                    note.as_str()
                        .is_some_and(|text| text.contains("did not include a preview-render step"))
                })),
            "the unsupported historical preview claim must remain explicitly corrected"
        );

        let checks = receipt
            .get("checks")
            .and_then(Value::as_array)
            .expect("CAD receipt must declare checks");
        assert!(!checks.is_empty());
        assert!(
            !checks.iter().any(|check| {
                check.get("id").and_then(Value::as_str) == Some("ODRH-CAD-PREVIEW-DEFAULT")
            }),
            "an unexecuted preview may not appear as a PASS"
        );
        for required_id in [
            "ODRH-CAD-NEGATIVE-APERTURE-ASSERTION",
            "ODRH-CAD-NEGATIVE-MOUNT-PITCH",
            "ODRH-CAD-NEGATIVE-PLATE-THICKNESS",
        ] {
            assert!(
                checks
                    .iter()
                    .any(|check| { check.get("id").and_then(Value::as_str) == Some(required_id) }),
                "historical CAD receipt omits evidenced negative control {required_id}"
            );
        }
        assert!(
            checks
                .iter()
                .all(|check| { check.get("status").and_then(Value::as_str) == Some("PASS") })
        );
        for check in checks {
            if let Some(hash) = check.get("output_sha256").and_then(Value::as_str) {
                assert_eq!(hash.len(), 64, "output SHA-256 must be 64 hex digits");
                assert!(
                    hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
                    "output SHA-256 must use hexadecimal digits"
                );
            }
        }
    }

    #[test]
    fn timestamp_and_calendar_parsers_accept_valid_rfc3339_cases() {
        assert!(is_valid_iso_date("2000-02-29"));
        assert!(is_valid_iso_date("2026-10-19"));
        assert!(!is_valid_iso_date("1900-02-29"));
        assert!(!is_valid_iso_date("2026-04-31"));

        assert!(is_valid_rfc3339_datetime("2026-10-18T10:00:00Z"));
        assert!(is_valid_rfc3339_datetime("2026-10-18t10:00:00.125+02:30"));
        assert!(is_valid_rfc3339_datetime("2000-02-29T23:59:60-00:00"));
    }

    #[test]
    fn timestamp_and_calendar_parsers_reject_malformed_values() {
        for value in [
            "2026-02-29T10:00:00Z",
            "2026-10-18T24:00:00Z",
            "2026-10-18T10:60:00Z",
            "2026-10-18T10:00:61Z",
            "2026-10-18T10:00:00",
            "2026-10-18T10:00:00+24:00",
            "2026-10-18T10:00:00+02:60",
            "2026-10-18T10:00:00.Z",
            "2026/10/18T10:00:00Z",
        ] {
            assert!(
                !is_valid_rfc3339_datetime(value),
                "malformed timestamp accepted: {value}"
            );
        }
        for value in ["2026-2-03", "2026-02-30", "2026-13-01", "not-a-date"] {
            assert!(
                !is_valid_iso_date(value),
                "malformed date accepted: {value}"
            );
        }
    }

    #[test]
    fn run_semantics_reject_invalid_acquisition_and_review_timestamps() {
        let mut bad_acquisition = parse_run();
        bad_acquisition["acquired_at"] = Value::String("2026-02-29T10:00:00Z".to_owned());
        assert!(
            validate_measurement_run_str(&encoded(&bad_acquisition), RUN_SCHEMA_TEXT)
                .expect_err("invalid acquisition date must fail")
                .iter()
                .any(|error| error.contains("acquired_at must be a valid RFC 3339"))
        );

        let mut bad_review = parse_run();
        bad_review["review"] = serde_json::json!({
            "reviewer_id":"synthetic reviewer",
            "review_method":"negative-control fixture",
            "reviewed_at":"2026-10-19T25:00:00Z",
            "decision":"NEEDS_MORE_EVIDENCE"
        });
        assert!(
            validate_measurement_run_str(&encoded(&bad_review), RUN_SCHEMA_TEXT)
                .expect_err("invalid review timestamp must fail")
                .iter()
                .any(|error| error.contains("review.reviewed_at must be a valid RFC 3339"))
        );
    }

    #[test]
    fn calibration_expiry_requires_a_real_calendar_date() {
        let mut record = parse_run();
        record["instruments"] = serde_json::json!([{
            "instrument_id":"ODRH-RUN-INS-POWER-001",
            "instrument_class":"POWER_METER",
            "manufacturer":"Example",
            "model":"Synthetic test-only instrument",
            "serial_or_stable_id":"NOT_A_PHYSICAL_INSTRUMENT",
            "calibration_state":"VALID",
            "calibration_record_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "calibration_valid_through":"2026-02-30",
            "supported_measurands":[{"measurand":"task-window input energy","unit":"J"}]
        }]);
        assert!(
            validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
                .expect_err("impossible calibration date must fail")
                .iter()
                .any(|error| error.contains("invalid calibration_valid_through date"))
        );
    }

    #[test]
    fn measurement_run_template_validates_but_remains_unexecuted() {
        let report = validate_measurement_run_str(RUN_TEXT, RUN_SCHEMA_TEXT)
            .expect("the unexecuted run template should satisfy the run contract");
        assert_eq!(report.record_id, "ODRH-RUN-V0-TEMPLATE");
        assert_eq!(report.record_state, "PLAN_ONLY");
        assert_eq!(report.observation_count, 4);
        assert_eq!(report.instrument_count, 0);
        assert_eq!(report.raw_artifact_count, 0);
    }

    #[test]
    fn plan_only_run_cannot_smuggle_a_measured_result() {
        let mut record = parse_run();
        record["observations"][0]["status"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["observations"][0]["value"] = serde_json::json!(10.0);
        record["observations"][0]["expanded_uncertainty"] = serde_json::json!(1.0);
        record["observations"][0]["minimum_decision_effect"] = serde_json::json!(3.0);
        record["observations"][0]["decision_capability"] = Value::String("CAPABLE".to_owned());
        let errors = validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect_err("a plan-only record cannot contain measured values");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("PLAN_ONLY record"))
        );
    }

    #[test]
    fn measurement_run_rejects_inconsistent_uncertainty_capability_claim() {
        let mut record = parse_run();
        record["record_state"] = Value::String("ACQUIRED_UNQUALIFIED".to_owned());
        record["physical_validation"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["acquired_at"] = Value::String("2026-10-18T10:00:00Z".to_owned());
        record["instruments"] = serde_json::json!([{
            "instrument_id":"ODRH-RUN-INS-POWER-001",
            "instrument_class":"POWER_METER",
            "manufacturer":"Example",
            "model":"Synthetic fixture",
            "serial_or_stable_id":"SYNTHETIC-NOT-A-PHYSICAL-INSTRUMENT",
            "calibration_state":"VALID",
            "calibration_record_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "calibration_valid_through":"2026-12-31",
            "calibration_record_artifact_id":"ODRH-ART-CAL-POWER-001",
            "capability_profile_artifact_id":"ODRH-ART-CAP-POWER-001",
            "capability_profile_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "capability_profile_status":"DECLARED_UNREVIEWED",
            "supported_measurands":[{"measurand":"task-window input energy","unit":"J"}]
        }]);
        record["raw_artifacts"] = serde_json::json!([
            {
                "artifact_id":"ODRH-ART-POWER-TRACE-001",
                "sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "media_type":"text/csv",
                "storage_uri":"https://example.invalid/synthetic-trace.csv",
                "byte_length":100,
                "role":"POWER_TRACE"
            },
            {
                "artifact_id":"ODRH-ART-CAL-POWER-001",
                "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "media_type":"application/pdf",
                "storage_uri":"https://example.invalid/power-calibration.pdf",
                "byte_length":300,
                "role":"INSTRUMENT_CALIBRATION"
            },
            {
                "artifact_id":"ODRH-ART-CAP-POWER-001",
                "sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "media_type":"application/json",
                "storage_uri":"https://example.invalid/power-capability.json",
                "byte_length":200,
                "role":"INSTRUMENT_CAPABILITY_PROFILE"
            }
        ]);
        record["observations"][0]["status"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["observations"][0]["value"] = serde_json::json!(10.0);
        record["observations"][0]["expanded_uncertainty"] = serde_json::json!(1.1);
        record["observations"][0]["minimum_decision_effect"] = serde_json::json!(3.0);
        record["observations"][0]["decision_capability"] = Value::String("CAPABLE".to_owned());
        record["observations"][0]["uncertainty_budget"] = serde_json::json!({
            "status":"CALCULATED_UNREVIEWED",
            "method":"LINEARIZED_COVARIANCE",
            "terms":[{
                "term_id":"POWER_METER",
                "label":"synthetic power-meter contribution",
                "standard_uncertainty":0.55,
                "sensitivity_coefficient":1.0,
                "notes":"synthetic test-only component"
            }],
            "covariance_terms":[],
            "combined_standard_uncertainty":0.55,
            "coverage_factor":2.0,
            "expanded_uncertainty":1.1,
            "notes":"synthetic test-only uncertainty budget"
        });
        record["observations"][0]["instrument_ids"] = serde_json::json!(["ODRH-RUN-INS-POWER-001"]);
        record["observations"][0]["raw_artifact_ids"] =
            serde_json::json!(["ODRH-ART-POWER-TRACE-001"]);

        let errors = validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect_err("uncertainty of 1.1 cannot pass MDE 3.0 under the one-third gate");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("claims CAPABLE despite uncertainty"))
        );
    }

    #[test]
    fn linearized_uncertainty_budget_recomputes_covariance_and_rejects_mutation() {
        let mut record = parse_run();
        let combined = (0.3_f64.powi(2) + 0.4_f64.powi(2) + 2.0 * 0.02).sqrt();
        let expanded = 2.0 * combined;
        record["record_state"] = Value::String("ACQUIRED_UNQUALIFIED".to_owned());
        record["physical_validation"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["acquired_at"] = Value::String("2026-10-18T10:00:00Z".to_owned());
        record["instruments"] = serde_json::json!([{
            "instrument_id":"ODRH-RUN-INS-POWER-001",
            "instrument_class":"POWER_METER",
            "manufacturer":"Example",
            "model":"Synthetic test reference",
            "serial_or_stable_id":"SYNTHETIC-POWER-001",
            "calibration_state":"VALID",
            "calibration_record_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "calibration_valid_through":"2026-12-31",
            "calibration_record_artifact_id":"ODRH-ART-CAL-POWER-001",
            "capability_profile_artifact_id":"ODRH-ART-CAP-POWER-001",
            "capability_profile_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "capability_profile_status":"DECLARED_UNREVIEWED",
            "supported_measurands":[{"measurand":"task-window input energy","unit":"J"}]
        }]);
        record["raw_artifacts"] = serde_json::json!([
            {
                "artifact_id":"ODRH-ART-POWER-TRACE-001",
                "sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "media_type":"text/csv",
                "storage_uri":"https://example.invalid/uncertainty-power.csv",
                "byte_length":100,
                "role":"POWER_TRACE"
            },
            {
                "artifact_id":"ODRH-ART-CAL-POWER-001",
                "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "media_type":"application/pdf",
                "storage_uri":"https://example.invalid/power-calibration.pdf",
                "byte_length":300,
                "role":"INSTRUMENT_CALIBRATION"
            },
            {
                "artifact_id":"ODRH-ART-CAP-POWER-001",
                "sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "media_type":"application/json",
                "storage_uri":"https://example.invalid/power-capability.json",
                "byte_length":200,
                "role":"INSTRUMENT_CAPABILITY_PROFILE"
            }
        ]);
        record["observations"][0]["status"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["observations"][0]["value"] = serde_json::json!(10.0);
        record["observations"][0]["expanded_uncertainty"] = serde_json::json!(expanded);
        record["observations"][0]["minimum_decision_effect"] = serde_json::json!(4.0);
        record["observations"][0]["decision_capability"] = Value::String("CAPABLE".to_owned());
        record["observations"][0]["instrument_ids"] = serde_json::json!(["ODRH-RUN-INS-POWER-001"]);
        record["observations"][0]["raw_artifact_ids"] =
            serde_json::json!(["ODRH-ART-POWER-TRACE-001"]);
        record["observations"][0]["uncertainty_budget"] = serde_json::json!({
            "status":"CALCULATED_UNREVIEWED",
            "method":"LINEARIZED_COVARIANCE",
            "terms":[
                {
                    "term_id":"METER_BIAS",
                    "label":"meter contribution",
                    "standard_uncertainty":0.3,
                    "sensitivity_coefficient":1.0,
                    "notes":"synthetic test term"
                },
                {
                    "term_id":"SESSION_REPEATABILITY",
                    "label":"session contribution",
                    "standard_uncertainty":0.4,
                    "sensitivity_coefficient":1.0,
                    "notes":"synthetic test term"
                }
            ],
            "covariance_terms":[{
                "term_a_id":"METER_BIAS",
                "term_b_id":"SESSION_REPEATABILITY",
                "covariance_contribution":0.02,
                "notes":"already sensitivity-projected covariance contribution"
            }],
            "combined_standard_uncertainty":combined,
            "coverage_factor":2.0,
            "expanded_uncertainty":expanded,
            "notes":"Synthetic known-answer budget only; not a physical uncertainty estimate."
        });

        validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect("well-formed linearized covariance budget should validate");

        let mut mismatched_unit = record.clone();
        mismatched_unit["instruments"][0]["supported_measurands"][0]["unit"] =
            Value::String("W".to_owned());
        assert!(
            validate_measurement_run_str(&encoded(&mismatched_unit), RUN_SCHEMA_TEXT)
                .expect_err("instrument capability must match the observation unit")
                .iter()
                .any(|error| {
                    error.contains(
                        "does not declare support for measurand task-window input energy in unit J",
                    )
                })
        );

        let mut reviewed = record.clone();
        reviewed["record_state"] = Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        reviewed["physical_validation"] = Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        reviewed["review"] = serde_json::json!({
            "reviewer_id":"synthetic independent reviewer",
            "review_method":"test-only schema/semantics fixture",
            "reviewed_at":"2026-10-19T09:00:00+02:00",
            "decision":"ACCEPTED_WITH_LIMITS"
        });
        reviewed["observations"][0]["status"] = Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        reviewed["observations"][0]["uncertainty_budget"]["status"] =
            Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        let errors = validate_measurement_run_str(&encoded(&reviewed), RUN_SCHEMA_TEXT).expect_err(
            "declared-but-unreviewed capability must not support independent result review",
        );
        assert!(
            errors.iter().any(|error| {
                error.contains("capability profile is not independently reviewed")
            })
        );

        reviewed["instruments"][0]["capability_profile_status"] =
            Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        validate_measurement_run_str(&encoded(&reviewed), RUN_SCHEMA_TEXT).expect(
            "the same synthetic record may pass the status gate once capability status is reviewed",
        );

        let mut bad_profile_digest = record.clone();
        let profile_index = bad_profile_digest["raw_artifacts"]
            .as_array()
            .expect("raw artifacts must be an array")
            .iter()
            .position(|artifact| {
                artifact.get("artifact_id").and_then(Value::as_str)
                    == Some("ODRH-ART-CAP-POWER-001")
            })
            .expect("capability-profile artifact must exist");
        bad_profile_digest["raw_artifacts"][profile_index]["sha256"] = Value::String(
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned(),
        );
        assert!(
            validate_measurement_run_str(&encoded(&bad_profile_digest), RUN_SCHEMA_TEXT)
                .expect_err("capability-profile artifact mutation must fail")
                .iter()
                .any(|error| error.contains("capability-profile digest does not match"))
        );

        let mut bad_calibration_role = record.clone();
        let calibration_index = bad_calibration_role["raw_artifacts"]
            .as_array()
            .expect("raw artifacts must be an array")
            .iter()
            .position(|artifact| {
                artifact.get("artifact_id").and_then(Value::as_str)
                    == Some("ODRH-ART-CAL-POWER-001")
            })
            .expect("calibration artifact must exist");
        bad_calibration_role["raw_artifacts"][calibration_index]["role"] =
            Value::String("OTHER".to_owned());
        assert!(
            validate_measurement_run_str(&encoded(&bad_calibration_role), RUN_SCHEMA_TEXT)
                .expect_err("wrong calibration-artifact role must fail")
                .iter()
                .any(|error| {
                    error.contains("calibration artifact ODRH-ART-CAL-POWER-001 has the wrong role")
                })
        );

        let mut mutated = record;
        mutated["observations"][0]["uncertainty_budget"]["covariance_terms"][0]["covariance_contribution"] =
            serde_json::json!(0.2);
        let errors = validate_measurement_run_str(&encoded(&mutated), RUN_SCHEMA_TEXT)
            .expect_err("mutated covariance must disagree with the frozen combined uncertainty");
        assert!(
            errors
                .iter()
                .any(|error| { error.contains("reported combined standard uncertainty") })
        );
    }

    #[test]
    fn not_evaluated_observation_can_retain_raw_evidence_without_derived_claim() {
        let mut record = parse_run();
        record["record_state"] = Value::String("ACQUIRED_UNQUALIFIED".to_owned());
        record["physical_validation"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["acquired_at"] = Value::String("2026-10-18T10:00:00Z".to_owned());
        record["instruments"] = serde_json::json!([
            {
                "instrument_id":"ODRH-RUN-INS-POWER-001",
                "instrument_class":"POWER_METER",
                "manufacturer":"Example",
                "model":"Synthetic reference meter",
                "serial_or_stable_id":"TEST-POWER-001",
                "calibration_state":"VALID",
                "calibration_record_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "calibration_valid_through":"2026-12-31",
                "calibration_record_artifact_id":"ODRH-ART-CAL-POWER-001",
                "capability_profile_artifact_id":"ODRH-ART-CAP-POWER-001",
                "capability_profile_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "capability_profile_status":"DECLARED_UNREVIEWED",
                "supported_measurands":[{"measurand":"task-window input energy","unit":"J"}]
            },
            {
                "instrument_id":"ODRH-RUN-INS-CAMERA-001",
                "instrument_class":"CAMERA",
                "manufacturer":"Example",
                "model":"Uncalibrated camera",
                "serial_or_stable_id":"TEST-CAMERA-001",
                "calibration_state":"UNKNOWN",
                "calibration_record_digest":null,
                "calibration_valid_through":null,
                "supported_measurands":[{"measurand":"relative image response","unit":"1"}]
            }
        ]);
        record["raw_artifacts"] = serde_json::json!([
            {
                "artifact_id":"ODRH-ART-POWER-001",
                "sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "media_type":"text/csv",
                "storage_uri":"https://example.invalid/power.csv",
                "byte_length":100,
                "role":"POWER_TRACE"
            },
            {
                "artifact_id":"ODRH-ART-CAMERA-001",
                "sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "media_type":"image/png",
                "storage_uri":"https://example.invalid/camera.png",
                "byte_length":200,
                "role":"OPTICAL_TRACE"
            },
            {
                "artifact_id":"ODRH-ART-CAL-POWER-001",
                "sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "media_type":"application/pdf",
                "storage_uri":"https://example.invalid/power-calibration.pdf",
                "byte_length":300,
                "role":"INSTRUMENT_CALIBRATION"
            },
            {
                "artifact_id":"ODRH-ART-CAP-POWER-001",
                "sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "media_type":"application/json",
                "storage_uri":"https://example.invalid/power-capability.json",
                "byte_length":200,
                "role":"INSTRUMENT_CAPABILITY_PROFILE"
            }
        ]);
        record["observations"][0]["status"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["observations"][0]["value"] = serde_json::json!(10.0);
        record["observations"][0]["expanded_uncertainty"] = serde_json::json!(0.5);
        record["observations"][0]["minimum_decision_effect"] = serde_json::json!(3.0);
        record["observations"][0]["decision_capability"] = Value::String("CAPABLE".to_owned());
        record["observations"][0]["uncertainty_budget"] = serde_json::json!({
            "status":"CALCULATED_UNREVIEWED",
            "method":"LINEARIZED_COVARIANCE",
            "terms":[{
                "term_id":"POWER_METER",
                "label":"synthetic power-meter contribution",
                "standard_uncertainty":0.25,
                "sensitivity_coefficient":1.0,
                "notes":"synthetic test-only component"
            }],
            "covariance_terms":[],
            "combined_standard_uncertainty":0.25,
            "coverage_factor":2.0,
            "expanded_uncertainty":0.5,
            "notes":"synthetic test-only uncertainty budget"
        });
        record["observations"][0]["instrument_ids"] = serde_json::json!(["ODRH-RUN-INS-POWER-001"]);
        record["observations"][0]["raw_artifact_ids"] = serde_json::json!(["ODRH-ART-POWER-001"]);
        record["observations"][1]["instrument_ids"] =
            serde_json::json!(["ODRH-RUN-INS-CAMERA-001"]);
        record["observations"][1]["raw_artifact_ids"] = serde_json::json!(["ODRH-ART-CAMERA-001"]);

        let report = validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect("unsupported metric may retain raw evidence without claiming a result");
        assert_eq!(report.record_state, "ACQUIRED_UNQUALIFIED");
        assert_eq!(report.raw_artifact_count, 4);
    }

    #[test]
    fn run_record_rejects_duplicate_observation_ids() {
        let mut record = parse_run();
        let first_id = record["observations"][0]["observation_id"].clone();
        record["observations"][1]["observation_id"] = first_id;
        assert!(
            validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
                .expect_err("duplicate observation identities must fail")
                .iter()
                .any(|error| error.contains("duplicate run observation ID"))
        );
    }

    #[test]
    fn run_record_rejects_duplicate_exclusion_ids() {
        let mut record = parse_run();
        record["exclusions"] = serde_json::json!([
            {
                "exclusion_id": "ODRH-EXCL-001",
                "observation_id": "ODRH-OBS-POWER-001",
                "reason": "synthetic negative control",
                "authority": "test fixture"
            },
            {
                "exclusion_id": "ODRH-EXCL-001",
                "observation_id": "ODRH-OBS-LUMINANCE-001",
                "reason": "synthetic duplicate identifier",
                "authority": "test fixture"
            }
        ]);
        assert!(
            validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
                .expect_err("duplicate exclusion identifiers must fail")
                .iter()
                .any(|error| error.contains("duplicate observation exclusion ID"))
        );
    }

    #[test]
    fn fixture_manifest_parameters_match_native_cad_defaults() {
        let manifest = parse_manifest();
        let fixture = manifest["modules"]
            .as_array()
            .expect("modules must be an array")
            .iter()
            .find(|module| module.get("id").and_then(Value::as_str) == Some("ODRH-FIXTURE-001"))
            .expect("fixture module must exist");
        let parameters = fixture["design_parameters"]
            .as_array()
            .expect("fixture design parameters must be an array");

        assert_eq!(parameters.len(), 10);
        for parameter in parameters {
            let name = parameter["name"]
                .as_str()
                .expect("parameter name must be a string");
            let value = parameter["value"]
                .as_f64()
                .expect("parameter value must be numeric");
            assert_eq!(
                parameter["status"].as_str(),
                Some("EXAMPLE_ONLY"),
                "CAD parameter {name} cannot be promoted by the specification manifest"
            );

            let prefix = format!("{name} =");
            let line = CAD_SOURCE
                .lines()
                .map(str::trim)
                .find(|line| line.starts_with(&prefix))
                .unwrap_or_else(|| panic!("CAD source is missing parameter {name}"));
            let assignment = line
                .split("//")
                .next()
                .expect("CAD parameter line must exist");
            let source_value = assignment
                .split_once('=')
                .expect("CAD parameter line must contain an assignment")
                .1
                .trim()
                .trim_end_matches(';')
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("CAD parameter {name} must have a literal default"));
            assert_eq!(
                value, source_value,
                "manifest default for {name} differs from native CAD source"
            );
        }
    }

    #[test]
    fn sha256_lower_hex_matches_standard_known_answer() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn frozen_v01_manifest_is_contract_valid_but_unqualified() {
        let report = validate_platform_str(MANIFEST_TEXT, SCHEMA_TEXT)
            .expect("the frozen specification manifest should satisfy its contract");
        assert_eq!(report.manifest_id, "ODRH-001");
        assert_eq!(report.schema_version, "0.1.0");
        assert_eq!(report.contract_state, "CONTRACT_VALID_UNQUALIFIED");
        assert_eq!(report.module_count, 8);
        assert_eq!(report.measurement_channel_count, 4);
        assert_eq!(report.reference_count, 14);
    }

    #[test]
    fn missing_module_id_is_rejected_by_schema() {
        let mut manifest = parse_manifest();
        assert!(
            manifest["modules"][0]
                .as_object_mut()
                .expect("module must be an object")
                .remove("id")
                .is_some()
        );
        assert!(validate_platform_str(&encoded(&manifest), SCHEMA_TEXT).is_err());
    }

    #[test]
    fn duplicate_module_identity_is_rejected() {
        let mut manifest = parse_manifest();
        let first_id = manifest["modules"][0]["id"].clone();
        manifest["modules"][1]["id"] = first_id;
        assert!(
            validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
                .expect_err("duplicate IDs must fail")
                .iter()
                .any(|error| error.contains("duplicate module ID"))
        );
    }

    #[test]
    fn unresolved_measurement_source_is_rejected() {
        let mut manifest = parse_manifest();
        manifest["measurement_channels"][0]["source_module_id"] =
            Value::String("ODRH-MISSING-999".to_owned());
        assert!(
            validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
                .expect_err("dangling source must fail")
                .iter()
                .any(|error| error.contains("unresolved source module"))
        );
    }

    #[test]
    fn unsupported_open_hardware_claim_is_rejected() {
        let mut manifest = parse_manifest();
        manifest["modules"][0]["current_openness_class"] =
            Value::String("PROJECT_CONTROLLED_OPEN".to_owned());
        assert!(
            validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
                .expect_err("open claim needs verified license and design source")
                .iter()
                .any(|error| error.contains("cannot claim PROJECT_CONTROLLED_OPEN"))
        );
    }

    #[test]
    fn specification_cannot_self_promote_physical_validation() {
        let mut manifest = parse_manifest();
        manifest["physical_validation"] = Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        assert!(
            validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
                .expect_err("physical evidence requires actual independent review")
                .iter()
                .any(|error| error.contains("physical_validation cannot be promoted"))
        );
    }

    #[test]
    fn specification_cannot_promote_benchmarks_or_gates() {
        let mut manifest = parse_manifest();
        manifest["benchmarks"][0]["evidence_status"] =
            Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        let errors = validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
            .expect_err("benchmark evidence cannot be promoted by editing the manifest");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("must remain NOT_MEASURED"))
        );

        let mut manifest = parse_manifest();
        manifest["gates"][0]["status"] = Value::String("PASSED_WITH_EVIDENCE".to_owned());
        assert!(
            validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
                .expect_err("gate status cannot be self-promoted")
                .iter()
                .any(|error| error.contains("must remain NOT_STARTED"))
        );
    }

    #[test]
    fn duplicate_reference_id_is_rejected() {
        let mut manifest = parse_manifest();
        let first_id = manifest["references"][0]["id"].clone();
        manifest["references"][1]["id"] = first_id;
        assert!(
            validate_platform_str(&encoded(&manifest), SCHEMA_TEXT)
                .expect_err("duplicate source IDs must fail")
                .iter()
                .any(|error| error.contains("duplicate reference ID"))
        );
    }
}
