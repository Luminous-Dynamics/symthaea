// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent Draft 2020-12 schema and instance audit for research contracts.
//!
//! Network/file reference retrieval is disabled by the isolated harness. This
//! tool refuses external references and validates the checked-in schemas and
//! instances entirely from local bytes.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::ExitCode};

const DRAFT_URI: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Serialize)]
struct SchemaReceipt {
    receipt_schema: &'static str,
    status: &'static str,
    qualification_scope: &'static str,
    schema_file: String,
    instance_file: String,
    schema_sha256: String,
    instance_sha256: String,
    declared_draft_2020_12: bool,
    schema_meta_valid: bool,
    external_references_absent: bool,
    validator_compiled: bool,
    instance_valid: bool,
    validation_errors: Vec<String>,
    claim_ceiling: &'static str,
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn find_external_reference(value: &Value) -> Option<String> {
    match value {
        Value::Object(object) => {
            for keyword in ["$ref", "$dynamicRef", "$recursiveRef"] {
                if let Some(Value::String(reference)) = object.get(keyword)
                    && !reference.starts_with('#')
                {
                    return Some(format!("{keyword}:{reference}"));
                }
            }
            object.values().find_map(find_external_reference)
        }
        Value::Array(values) => values.iter().find_map(find_external_reference),
        _ => None,
    }
}

fn file_name(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("<non-utf8>")
        .to_owned()
}

fn audit_schema(
    schema_bytes: &[u8],
    instance_bytes: &[u8],
    schema_file: String,
    instance_file: String,
) -> SchemaReceipt {
    let mut validation_errors = Vec::new();
    let mut declared_draft_2020_12 = false;
    let mut schema_meta_valid = false;
    let mut external_references_absent = false;
    let mut validator_compiled = false;
    let mut instance_valid = false;

    let schema: Option<Value> = match serde_json::from_slice(schema_bytes) {
        Ok(value) => Some(value),
        Err(error) => {
            validation_errors.push(format!("schema_json_parse_error:{error}"));
            None
        }
    };
    let instance: Option<Value> = match serde_json::from_slice(instance_bytes) {
        Ok(value) => Some(value),
        Err(error) => {
            validation_errors.push(format!("instance_json_parse_error:{error}"));
            None
        }
    };

    if let Some(schema) = schema.as_ref() {
        declared_draft_2020_12 = schema.get("$schema").and_then(Value::as_str) == Some(DRAFT_URI);
        if !declared_draft_2020_12 {
            validation_errors.push("schema_draft_uri_mismatch".to_owned());
        }

        schema_meta_valid = jsonschema::draft202012::meta::is_valid(schema);
        if !schema_meta_valid {
            validation_errors.push("schema_fails_draft_2020_12_meta_schema".to_owned());
        }

        external_references_absent = find_external_reference(schema).is_none();
        if !external_references_absent {
            validation_errors
                .push("external_schema_reference_refused_by_local_only_auditor".to_owned());
        }

        if declared_draft_2020_12 && schema_meta_valid && external_references_absent {
            match jsonschema::draft202012::options()
                .should_validate_formats(true)
                .build(schema)
            {
                Ok(validator) => {
                    validator_compiled = true;
                    if let Some(instance) = instance.as_ref() {
                        let errors = validator
                            .iter_errors(instance)
                            .map(|error| {
                                format!("instance_invalid_at_{}:{error}", error.instance_path())
                            })
                            .collect::<Vec<_>>();
                        instance_valid = errors.is_empty();
                        validation_errors.extend(errors);
                    }
                }
                Err(error) => validation_errors.push(format!("schema_compile_error:{error}")),
            }
        }
    }

    let passed = declared_draft_2020_12
        && schema_meta_valid
        && external_references_absent
        && validator_compiled
        && instance_valid
        && validation_errors.is_empty();

    SchemaReceipt {
        receipt_schema: "json-schema-contract-receipt-v1",
        status: if passed { "PASS" } else { "FAIL" },
        qualification_scope: "draft_2020_12_meta_validation_and_local_instance_validation",
        schema_file,
        instance_file,
        schema_sha256: hex_sha256(schema_bytes),
        instance_sha256: hex_sha256(instance_bytes),
        declared_draft_2020_12,
        schema_meta_valid,
        external_references_absent,
        validator_compiled,
        instance_valid,
        validation_errors,
        claim_ceiling: "This receipt validates a local JSON Schema and instance with the pinned jsonschema library's Draft 2020-12 implementation and format validation. It does not authenticate source files, test every official conformance-suite case, validate domain semantics beyond the schema, or prove scientific validity.",
    }
}

fn parse_args() -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let mut schema = None;
    let mut instance = None;
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing_value_for_{flag}"))?;
        match flag.as_str() {
            "--schema" => schema = Some(PathBuf::from(value)),
            "--instance" => instance = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown_argument_{flag}")),
        }
    }

    Ok((
        schema.ok_or_else(|| "missing_--schema".to_owned())?,
        instance.ok_or_else(|| "missing_--instance".to_owned())?,
        output.ok_or_else(|| "missing_--out".to_owned())?,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (schema_path, instance_path, output_path) = parse_args()?;
    let schema_bytes =
        fs::read(&schema_path).map_err(|error| format!("schema_read_failed:{error}"))?;
    let instance_bytes =
        fs::read(&instance_path).map_err(|error| format!("instance_read_failed:{error}"))?;
    let receipt = audit_schema(
        &schema_bytes,
        &instance_bytes,
        file_name(&schema_path),
        file_name(&instance_path),
    );
    let passed = receipt.status == "PASS";
    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("receipt_serialization_failed:{error}"))?;
    fs::write(&output_path, serialized).map_err(|error| format!("receipt_write_failed:{error}"))?;
    if passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("json-schema-contract-audit:{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS_SCHEMA: &[u8] =
        include_bytes!("../../../../docs/research/beverage-engineering-scenarios-v1.schema.json");
    const CORPUS: &[u8] =
        include_bytes!("../../../../docs/research/beverage-engineering-scenarios-v1.json");

    #[test]
    fn checked_in_synthetic_corpus_passes_draft_2020_12_validation() {
        let receipt = audit_schema(
            CORPUS_SCHEMA,
            CORPUS,
            "beverage-engineering-scenarios-v1.schema.json".to_owned(),
            "beverage-engineering-scenarios-v1.json".to_owned(),
        );
        assert_eq!(receipt.status, "PASS", "{:?}", receipt.validation_errors);
        assert!(receipt.schema_meta_valid);
        assert!(receipt.external_references_absent);
        assert!(receipt.validator_compiled);
        assert!(receipt.instance_valid);
        assert_eq!(receipt.schema_sha256.len(), 64);
        assert_eq!(receipt.instance_sha256.len(), 64);
    }

    #[test]
    fn missing_required_field_fails_instance_validation() {
        let schema =
            br#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["id"]}"#;
        let instance = br#"{"other":"value"}"#;
        let receipt = audit_schema(
            schema,
            instance,
            "schema.json".to_owned(),
            "data.json".to_owned(),
        );
        assert_eq!(receipt.status, "FAIL");
        assert!(receipt.schema_meta_valid);
        assert!(!receipt.instance_valid);
    }

    #[test]
    fn malformed_schema_fails_closed() {
        let receipt = audit_schema(
            b"{",
            br#"{}"#,
            "schema.json".to_owned(),
            "data.json".to_owned(),
        );
        assert_eq!(receipt.status, "FAIL");
        assert!(
            receipt
                .validation_errors
                .iter()
                .any(|error| error.starts_with("schema_json_parse_error:"))
        );
    }

    #[test]
    fn unsupported_schema_draft_fails_closed() {
        let schema = br#"{"$schema":"http://json-schema.org/draft-07/schema#","type":"object"}"#;
        let receipt = audit_schema(
            schema,
            br#"{}"#,
            "schema.json".to_owned(),
            "data.json".to_owned(),
        );
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.declared_draft_2020_12);
    }

    #[test]
    fn external_references_are_refused_without_network_access() {
        let schema =
            br#"{"$schema":"https://json-schema.org/draft/2020-12/schema","$ref":"https://example.invalid/schema.json"}"#;
        let receipt = audit_schema(
            schema,
            br#"{}"#,
            "schema.json".to_owned(),
            "data.json".to_owned(),
        );
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.external_references_absent);
        assert!(
            receipt
                .validation_errors
                .iter()
                .any(|error| error.contains("external_schema_reference_refused"))
        );
    }

    #[test]
    fn invalid_meta_schema_fails_closed() {
        let schema =
            br#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"not-a-valid-type"}"#;
        let receipt = audit_schema(
            schema,
            br#"{}"#,
            "schema.json".to_owned(),
            "data.json".to_owned(),
        );
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.schema_meta_valid);
    }
}
