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

    validate_unique_ids(instruments, "run instrument", &mut errors);
    validate_unique_ids(observations, "run observation", &mut errors);
    validate_unique_ids(artifacts, "raw artifact", &mut errors);
    validate_unique_field(exclusions, "observation exclusion", "exclusion_id", &mut errors);

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

    for instrument in instruments {
        let instrument_id = instrument
            .get("instrument_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if instrument.get("calibration_state").and_then(Value::as_str) == Some("VALID") {
            if instrument
                .get("calibration_record_digest")
                .and_then(Value::as_str)
                .is_none()
                || instrument
                    .get("calibration_valid_through")
                    .and_then(Value::as_str)
                    .is_none()
            {
                errors.push(format!(
                    "instrument {instrument_id} cannot claim VALID calibration without record digest and expiry"
                ));
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
            if observation.get("value").is_some_and(|value| !value.is_null())
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
                if instrument.get("calibration_valid_through").and_then(Value::as_str)
                    .zip(run_date)
                    .is_some_and(|(expiry, date)| expiry < date)
                {
                    errors.push(format!(
                        "measured observation {observation_id} uses instrument {id} after its calibration expiry"
                    ));
                }
                let supports_measurand = instrument
                    .get("supported_measurands")
                    .and_then(Value::as_array)
                    .is_some_and(|measurands| {
                        measurands
                            .iter()
                            .any(|item| item.as_str() == Some(measurand))
                    });
                if !supports_measurand {
                    errors.push(format!(
                        "instrument {id} does not declare support for measurand {measurand}"
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
                    "ABORTED record with no acquired data must not claim an acquisition timestamp".to_owned(),
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
                    if manifest.get("status").and_then(Value::as_str)
                        == Some("SPECIFICATION_ONLY")
                        && parameter.get("status").and_then(Value::as_str)
                            != Some("EXAMPLE_ONLY")
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

fn validate_unique_field(
    records: &[Value],
    category: &str,
    field: &str,
    errors: &mut Vec<String>,
) {
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
    use super::{sha256_hex, validate_measurement_run_str, validate_platform_str};
    use serde_json::Value;

    const MANIFEST_TEXT: &str =
        include_str!("../../../docs/engineering/open-display-reference-platform-v0.1.json");
    const SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/open-display-reference-platform.schema.json");
    const CAD_SOURCE: &str =
        include_str!("../../../hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad");
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
        assert!(errors.iter().any(|error| error.contains("PLAN_ONLY record")));
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
            "supported_measurands":["task-window input energy"]
        }]);
        record["raw_artifacts"] = serde_json::json!([{
            "artifact_id":"ODRH-ART-POWER-TRACE-001",
            "sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "media_type":"text/csv",
            "storage_uri":"https://example.invalid/synthetic-trace.csv",
            "byte_length":100,
            "role":"POWER_TRACE"
        }]);
        record["observations"][0]["status"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["observations"][0]["value"] = serde_json::json!(10.0);
        record["observations"][0]["expanded_uncertainty"] = serde_json::json!(1.1);
        record["observations"][0]["minimum_decision_effect"] = serde_json::json!(3.0);
        record["observations"][0]["decision_capability"] = Value::String("CAPABLE".to_owned());
        record["observations"][0]["instrument_ids"] =
            serde_json::json!(["ODRH-RUN-INS-POWER-001"]);
        record["observations"][0]["raw_artifact_ids"] =
            serde_json::json!(["ODRH-ART-POWER-TRACE-001"]);

        let errors = validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect_err("uncertainty of 1.1 cannot pass MDE 3.0 under the one-third gate");
        assert!(errors
            .iter()
            .any(|error| error.contains("claims CAPABLE despite uncertainty")));
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
                "supported_measurands":["task-window input energy"]
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
                "supported_measurands":["relative image response"]
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
            }
        ]);
        record["observations"][0]["status"] = Value::String("MEASURED_UNQUALIFIED".to_owned());
        record["observations"][0]["value"] = serde_json::json!(10.0);
        record["observations"][0]["expanded_uncertainty"] = serde_json::json!(0.5);
        record["observations"][0]["minimum_decision_effect"] = serde_json::json!(3.0);
        record["observations"][0]["decision_capability"] = Value::String("CAPABLE".to_owned());
        record["observations"][0]["instrument_ids"] = serde_json::json!(["ODRH-RUN-INS-POWER-001"]);
        record["observations"][0]["raw_artifact_ids"] = serde_json::json!(["ODRH-ART-POWER-001"]);
        record["observations"][1]["instrument_ids"] =
            serde_json::json!(["ODRH-RUN-INS-CAMERA-001"]);
        record["observations"][1]["raw_artifact_ids"] =
            serde_json::json!(["ODRH-ART-CAMERA-001"]);

        let report = validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect("unsupported metric may retain raw evidence without claiming a result");
        assert_eq!(report.record_state, "ACQUIRED_UNQUALIFIED");
        assert_eq!(report.raw_artifact_count, 2);
    }

    #[test]
    fn run_record_rejects_duplicate_observation_ids() {
        let mut record = parse_run();
        let first_id = record["observations"][0]["observation_id"].clone();
        record["observations"][1]["observation_id"] = first_id;
        assert!(validate_measurement_run_str(&encoded(&record), RUN_SCHEMA_TEXT)
            .expect_err("duplicate observation identities must fail")
            .iter()
            .any(|error| error.contains("duplicate run observation ID")));
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
        assert_eq!(report.reference_count, 10);
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
