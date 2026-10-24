// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation for run-generated OpenSCAD geometry receipts.
//!
//! This validates source binding and receipt consistency, not the truth of an
//! uploaded file or the physical suitability of the CAD model.

use super::{EXPECTED_SCHEMA_DRAFT, approximately_equal, count_array, sha256_hex};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CadRuntimeReceiptValidationReport {
    pub validation_id: String,
    pub output_count: usize,
    pub negative_control_count: usize,
    pub receipt_sha256: String,
    pub schema_sha256: String,
    pub source_sha256: String,
}

impl CadRuntimeReceiptValidationReport {
    /// Stable validation result with a geometry-only claim ceiling.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "verifier": "open-display-platform-verify",
            "verifier_version": env!("CARGO_PKG_VERSION"),
            "validation_id": self.validation_id,
            "contract_state": "CAD_RECEIPT_CONTRACT_VALID",
            "output_count": self.output_count,
            "negative_control_count": self.negative_control_count,
            "receipt_sha256": self.receipt_sha256,
            "schema_sha256": self.schema_sha256,
            "source_sha256": self.source_sha256,
            "physical_validation": "NOT_PERFORMED",
            "claim_ceiling": "Receipt metadata and source binding validated. This is not a physical print, fit, strength, optical, calibration, or metrology qualification."
        })
    }
}

/// Validate a run-generated CAD receipt against its schema and exact source.
///
/// # Errors
///
/// Returns schema or semantic failures. A valid receipt only establishes that
/// declared geometry-export checks are represented consistently; it does not
/// prove the workflow ran or independently certify the receipt's claims.
pub fn validate_cad_runtime_receipt_str(
    receipt_text: &str,
    schema_text: &str,
    cad_source_text: &str,
) -> Result<CadRuntimeReceiptValidationReport, Vec<String>> {
    let schema: Value = match serde_json::from_str(schema_text) {
        Ok(value) => value,
        Err(error) => {
            return Err(vec![format!(
                "CAD receipt schema JSON parse error: {error}"
            )]);
        }
    };
    let receipt: Value = match serde_json::from_str(receipt_text) {
        Ok(value) => value,
        Err(error) => {
            return Err(vec![format!(
                "CAD runtime receipt JSON parse error: {error}"
            )]);
        }
    };

    let mut errors = Vec::new();
    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_DRAFT) {
        errors.push(format!(
            "CAD receipt schema must declare {EXPECTED_SCHEMA_DRAFT}"
        ));
    }
    match jsonschema::draft202012::new(&schema) {
        Ok(validator) => {
            for error in validator.iter_errors(&receipt) {
                errors.push(format!("CAD receipt JSON Schema violation: {error}"));
            }
        }
        Err(error) => errors.push(format!("CAD receipt schema compilation failed: {error}")),
    }
    errors.extend(validate_semantics(&receipt, cad_source_text));
    errors.sort();
    errors.dedup();
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(CadRuntimeReceiptValidationReport {
        validation_id: receipt
            .get("validation_id")
            .and_then(Value::as_str)
            .expect("schema validates validation_id")
            .to_owned(),
        output_count: count_array(&receipt, "outputs"),
        negative_control_count: count_array(&receipt, "negative_controls"),
        receipt_sha256: sha256_hex(receipt_text.as_bytes()),
        schema_sha256: sha256_hex(schema_text.as_bytes()),
        source_sha256: sha256_hex(cad_source_text.as_bytes()),
    })
}

fn validate_semantics(receipt: &Value, cad_source_text: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let expected_source_sha256 = sha256_hex(cad_source_text.as_bytes());
    if receipt.pointer("/source/sha256").and_then(Value::as_str)
        != Some(expected_source_sha256.as_str())
    {
        errors
            .push("CAD runtime receipt source SHA-256 does not match the native source".to_owned());
    }

    let repository = receipt
        .pointer("/execution/repository")
        .and_then(Value::as_str)
        .unwrap_or("");
    let run_id = receipt
        .pointer("/execution/workflow_run_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let expected_run_url = format!("https://github.com/{repository}/actions/runs/{run_id}");
    if receipt
        .pointer("/execution/workflow_run_url")
        .and_then(Value::as_str)
        != Some(expected_run_url.as_str())
    {
        errors.push(
            "CAD runtime receipt workflow URL does not match repository/run identity".to_owned(),
        );
    }

    let version_output = receipt
        .pointer("/tool/version_output")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !version_output.trim().starts_with("OpenSCAD version ") {
        errors.push("CAD runtime receipt must name OpenSCAD version output".to_owned());
    }

    validate_outputs(receipt, &mut errors);
    validate_negative_controls(receipt, &mut errors);

    if receipt.get("physical_validation").and_then(Value::as_str) != Some("NOT_PERFORMED") {
        errors.push("CAD runtime receipt cannot promote physical validation".to_owned());
    }
    if receipt.get("claim_boundary").and_then(Value::as_str)
        != Some("GEOMETRY_EXPORT_CHECKED_NOT_PHYSICALLY_QUALIFIED")
    {
        errors.push("CAD runtime receipt claim boundary is invalid".to_owned());
    }
    errors
}

fn validate_outputs(receipt: &Value, errors: &mut Vec<String>) {
    let Some(outputs) = receipt.get("outputs").and_then(Value::as_array) else {
        errors.push("CAD runtime receipt outputs must be an array".to_owned());
        return;
    };

    let mut names = BTreeSet::new();
    for output in outputs {
        let name = output
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if !names.insert(name.to_owned()) {
            errors.push(format!("CAD runtime receipt repeats output {name}"));
        }

        let expected_bounds: Option<[f64; 3]> = match name {
            "odrh-default.stl" => Some([80.0, 5.0, 80.0]),
            "odrh-alternate.stl" => Some([100.0, 5.0, 60.0]),
            _ => None,
        };
        match (
            expected_bounds,
            output.get("expected_bounds_mm").and_then(Value::as_array),
        ) {
            (Some(expected), Some(actual)) if actual.len() == 3 => {
                for (index, (actual_value, expected_value)) in
                    actual.iter().zip(expected.iter()).enumerate()
                {
                    if actual_value
                        .as_f64()
                        .is_none_or(|value| !approximately_equal(value, *expected_value))
                    {
                        errors.push(format!(
                            "CAD runtime receipt output {name} has incorrect expected bound on axis {index}"
                        ));
                    }
                }
            }
            (Some(_), _) => errors.push(format!(
                "CAD runtime receipt output {name} must declare three expected bounds"
            )),
            (None, _) => errors.push(format!(
                "CAD runtime receipt contains unsupported mesh output name {name}"
            )),
        }

        if name == "odrh-alternate.stl" {
            let width = output
                .pointer("/override/plate_width_mm")
                .and_then(Value::as_f64);
            let height = output
                .pointer("/override/plate_height_mm")
                .and_then(Value::as_f64);
            if width.is_none_or(|value| !approximately_equal(value, 100.0))
                || height.is_none_or(|value| !approximately_equal(value, 60.0))
            {
                errors.push(
                    "alternate CAD output must bind the declared 100 x 60 mm override".to_owned(),
                );
            }
        } else if output.get("override").is_some() {
            errors.push(format!(
                "default CAD output {name} must not declare an override"
            ));
        }
    }

    let expected_names = BTreeSet::from([
        "odrh-default.stl".to_owned(),
        "odrh-alternate.stl".to_owned(),
    ]);
    if names != expected_names {
        errors.push(
            "CAD runtime receipt must contain exactly the default and alternate STL outputs"
                .to_owned(),
        );
    }
}

fn validate_negative_controls(receipt: &Value, errors: &mut Vec<String>) {
    const EXPECTED: [(&str, f64, &str); 3] = [
        (
            "sensor_aperture_d_mm",
            100.0,
            "Sensor aperture overlaps the diagonal mounting-hole envelope",
        ),
        (
            "mount_pitch_x_mm",
            100.0,
            "Horizontal mounting-hole grid violates edge margin",
        ),
        (
            "plate_thickness_mm",
            0.5,
            "Counterbore depth must be smaller than plate thickness",
        ),
    ];

    let Some(controls) = receipt.get("negative_controls").and_then(Value::as_array) else {
        errors.push("CAD runtime receipt negative_controls must be an array".to_owned());
        return;
    };

    let mut parameters = BTreeSet::new();
    for control in controls {
        let parameter = control
            .get("parameter")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        if !parameters.insert(parameter.to_owned()) {
            errors.push(format!(
                "CAD receipt repeats negative-control parameter {parameter}"
            ));
        }
    }

    for (parameter, expected_override, expected_failure) in EXPECTED {
        let matching = controls
            .iter()
            .find(|control| control.get("parameter").and_then(Value::as_str) == Some(parameter));
        match matching {
            Some(control) => {
                if control
                    .get("override")
                    .and_then(Value::as_f64)
                    .is_none_or(|value| !approximately_equal(value, expected_override))
                {
                    errors.push(format!(
                        "CAD negative control {parameter} has an incorrect override"
                    ));
                }
                if control.get("expected_failure").and_then(Value::as_str) != Some(expected_failure)
                {
                    errors.push(format!(
                        "CAD negative control {parameter} has an unexpected failure condition"
                    ));
                }
            }
            None => errors.push(format!(
                "CAD runtime receipt omits negative control {parameter}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_cad_runtime_receipt_str;
    use crate::sha256_hex;
    use serde_json::{Value, json};

    const CAD_SOURCE: &str = include_str!(
        "../../../hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad"
    );
    const SCHEMA_TEXT: &str = include_str!(
        "../../../hardware/open-display-reference/validation/odrh-cad-runtime-receipt.schema.json"
    );

    fn synthetic_receipt() -> Value {
        json!({
            "schema_version": "0.1.0",
            "validation_id": "ODRH-CAD-RUN-001",
            "claim_boundary": "GEOMETRY_EXPORT_CHECKED_NOT_PHYSICALLY_QUALIFIED",
            "source": {
                "path": "hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad",
                "sha256": sha256_hex(CAD_SOURCE.as_bytes())
            },
            "tool": {
                "name": "OpenSCAD",
                "version_output": "OpenSCAD version 2021.01\n"
            },
            "execution": {
                "repository": "Luminous-Dynamics/symthaea",
                "commit_sha": "0123456789abcdef0123456789abcdef01234567",
                "workflow_run_id": "12345",
                "workflow_run_attempt": "1",
                "workflow_run_url": "https://github.com/Luminous-Dynamics/symthaea/actions/runs/12345"
            },
            "outputs": [
                {
                    "name": "odrh-default.stl",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "expected_bounds_mm": [80, 5, 80]
                },
                {
                    "name": "odrh-alternate.stl",
                    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "override": {"plate_width_mm": 100, "plate_height_mm": 60},
                    "expected_bounds_mm": [100, 5, 60]
                }
            ],
            "negative_controls": [
                {
                    "parameter": "sensor_aperture_d_mm",
                    "override": 100,
                    "expected_failure": "Sensor aperture overlaps the diagonal mounting-hole envelope"
                },
                {
                    "parameter": "mount_pitch_x_mm",
                    "override": 100,
                    "expected_failure": "Horizontal mounting-hole grid violates edge margin"
                },
                {
                    "parameter": "plate_thickness_mm",
                    "override": 0.5,
                    "expected_failure": "Counterbore depth must be smaller than plate thickness"
                }
            ],
            "physical_validation": "NOT_PERFORMED"
        })
    }

    fn encoded(value: &Value) -> String {
        serde_json::to_string(value).expect("synthetic receipt must serialize")
    }

    #[test]
    fn synthetic_receipt_validates_and_preserves_geometry_only_claim_ceiling() {
        let receipt = encoded(&synthetic_receipt());
        let report = validate_cad_runtime_receipt_str(&receipt, SCHEMA_TEXT, CAD_SOURCE)
            .expect("synthetic run receipt should conform");
        assert_eq!(report.validation_id, "ODRH-CAD-RUN-001");
        assert_eq!(report.output_count, 2);
        assert_eq!(report.negative_control_count, 3);
        assert_eq!(report.source_sha256, sha256_hex(CAD_SOURCE.as_bytes()));
        assert!(
            report.to_json()["claim_ceiling"]
                .as_str()
                .unwrap()
                .contains("not a physical print")
        );
    }

    #[test]
    fn receipt_rejects_native_source_digest_drift() {
        let mut receipt = synthetic_receipt();
        receipt["source"]["sha256"] = Value::String(
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
        );
        assert!(
            validate_cad_runtime_receipt_str(&encoded(&receipt), SCHEMA_TEXT, CAD_SOURCE)
                .expect_err("mutated native-source digest must fail")
                .iter()
                .any(|error| error.contains("source SHA-256 does not match"))
        );
    }

    #[test]
    fn receipt_rejects_run_url_and_geometry_mismatch() {
        let mut receipt = synthetic_receipt();
        receipt["execution"]["workflow_run_url"] =
            Value::String("https://github.com/example/wrong/actions/runs/123".to_owned());
        receipt["outputs"][0]["expected_bounds_mm"][0] = json!(79);
        let errors = validate_cad_runtime_receipt_str(&encoded(&receipt), SCHEMA_TEXT, CAD_SOURCE)
            .expect_err("wrong execution URL and geometry bounds must fail");
        assert!(errors.iter().any(|error| error.contains("workflow URL")));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("incorrect expected bound"))
        );
    }

    #[test]
    fn receipt_cannot_promote_physical_qualification() {
        let mut receipt = synthetic_receipt();
        receipt["physical_validation"] = Value::String("INDEPENDENTLY_REVIEWED".to_owned());
        let errors = validate_cad_runtime_receipt_str(&encoded(&receipt), SCHEMA_TEXT, CAD_SOURCE)
            .expect_err("the schema const must reject physical-status promotion");
        assert!(
            !errors.is_empty(),
            "a physical qualification claim must make the receipt invalid"
        );
    }

    #[test]
    fn receipt_cannot_drop_required_negative_controls() {
        let mut receipt = synthetic_receipt();
        receipt["negative_controls"]
            .as_array_mut()
            .expect("negative controls are an array")
            .remove(0);
        let errors = validate_cad_runtime_receipt_str(&encoded(&receipt), SCHEMA_TEXT, CAD_SOURCE)
            .expect_err("the schema must reject an incomplete negative-control set");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("negative_controls")),
            "missing negative control should be reported by the schema"
        );
    }
}
