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
    let mut seen = BTreeSet::new();
    for record in records {
        let id = record
            .get("id")
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
    use super::{sha256_hex, validate_platform_str};
    use serde_json::Value;

    const MANIFEST_TEXT: &str =
        include_str!("../../../docs/engineering/open-display-reference-platform-v0.1.json");
    const SCHEMA_TEXT: &str =
        include_str!("../../../docs/engineering/open-display-reference-platform.schema.json");
    const CAD_SOURCE: &str =
        include_str!("../../../hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad");

    fn parse_manifest() -> Value {
        serde_json::from_str(MANIFEST_TEXT).expect("fixture manifest must parse")
    }

    fn encoded(value: &Value) -> String {
        serde_json::to_string(value).expect("mutated fixture must serialize")
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
