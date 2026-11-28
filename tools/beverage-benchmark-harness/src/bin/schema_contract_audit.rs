//! Offline Draft 2020-12 schema and instance auditor for the beverage corpus.
//!
//! This executable validates the schema document against the bundled Draft 2020-12
//! meta-schema, then validates the corpus instance. Remote and file reference
//! retrieval is disabled; validation must be self-contained and reproducible.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const DRAFT_2020_12_URI: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Clone)]
struct Args {
    schema: PathBuf,
    instance: PathBuf,
    expected_head: String,
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct AuditReceipt {
    receipt_schema_version: &'static str,
    auditor: &'static str,
    status: &'static str,
    expected_head: String,
    schema_uri: String,
    schema_sha256: String,
    instance_sha256: String,
    schema_meta_valid: bool,
    validator_built: bool,
    instance_valid: Option<bool>,
    validation_errors: Vec<String>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut schema = None;
    let mut instance = None;
    let mut expected_head = None;
    let mut output = None;
    let mut args = arguments.into_iter();

    while let Some(argument) = args.next() {
        let value = || format!("missing value after {argument}");
        match argument.as_str() {
            "--schema" => schema = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--instance" => instance = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--expected-head" => expected_head = Some(args.next().ok_or_else(value)?),
            "--out" => output = Some(PathBuf::from(args.next().ok_or_else(value)?)),
            "--help" | "-h" => {
                return Err(
                    "usage: schema_contract_audit --schema PATH --instance PATH \
                     --expected-head GIT_SHA [--out PATH]"
                        .to_owned(),
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let expected_head = expected_head.ok_or("--expected-head is required")?;
    if !matches!(expected_head.len(), 40 | 64)
        || !expected_head.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("--expected-head must be a 40- or 64-character hexadecimal Git object ID".into());
    }

    Ok(Args {
        schema: schema.ok_or("--schema is required")?,
        instance: instance.ok_or("--instance is required")?,
        expected_head,
        output,
    })
}

fn validate_values(schema: &Value, instance: &Value) -> Result<Vec<String>, String> {
    let schema_uri = schema
        .get("$schema")
        .and_then(Value::as_str)
        .ok_or("schema must declare a string $schema dialect URI")?;
    if schema_uri != DRAFT_2020_12_URI {
        return Err(format!(
            "unsupported schema dialect {schema_uri:?}; expected {DRAFT_2020_12_URI:?}"
        ));
    }

    jsonschema::draft202012::meta::validate(schema)
        .map_err(|error| format!("Draft 2020-12 meta-schema validation failed: {error}"))?;

    // Offline mode permits local fragments and bundled known meta-schemas but
    // refuses network/file retrieval for unresolved references.
    let validator = jsonschema::draft202012::options()
        .offline()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("could not build offline Draft 2020-12 validator: {error}"))?;

    Ok(validator
        .iter_errors(instance)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect())
}

fn make_receipt(
    args: &Args,
    schema_bytes: &[u8],
    instance_bytes: &[u8],
) -> Result<AuditReceipt, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let instance: Value = serde_json::from_slice(instance_bytes)
        .map_err(|error| format!("instance JSON parse failed: {error}"))?;

    let schema_uri = schema
        .get("$schema")
        .and_then(Value::as_str)
        .unwrap_or("MISSING_OR_INVALID")
        .to_owned();

    let (schema_meta_valid, validator_built, instance_valid, validation_errors) =
        match validate_values(&schema, &instance) {
            Ok(errors) => (true, true, Some(errors.is_empty()), errors),
            Err(error) => {
                let meta_valid = jsonschema::draft202012::meta::validate(&schema).is_ok()
                    && schema_uri == DRAFT_2020_12_URI;
                (meta_valid, false, None, vec![error])
            }
        };

    let status = if schema_meta_valid && validator_built && instance_valid == Some(true) {
        "PASS"
    } else {
        "FAIL"
    };

    Ok(AuditReceipt {
        receipt_schema_version: "0.1.0",
        auditor: "jsonschema 0.59.0 / Draft 2020-12 / offline",
        status,
        expected_head: args.expected_head.clone(),
        schema_uri,
        schema_sha256: sha256(schema_bytes),
        instance_sha256: sha256(instance_bytes),
        schema_meta_valid,
        validator_built,
        instance_valid,
        validation_errors,
    })
}

fn write_receipt(receipt: &AuditReceipt, output: Option<&Path>) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(receipt)
        .map_err(|error| format!("receipt serialization failed: {error}"))?;
    if let Some(path) = output {
        fs::write(path, format!("{serialized}\n"))
            .map_err(|error| format!("could not write receipt {}: {error}", path.display()))?;
    }
    println!("{serialized}");
    Ok(())
}

fn run() -> Result<bool, String> {
    let args = parse_args(env::args().skip(1))?;
    let schema_bytes = fs::read(&args.schema)
        .map_err(|error| format!("could not read schema {}: {error}", args.schema.display()))?;
    let instance_bytes = fs::read(&args.instance)
        .map_err(|error| format!("could not read instance {}: {error}", args.instance.display()))?;

    let receipt = make_receipt(&args, &schema_bytes, &instance_bytes)?;
    write_receipt(&receipt, args.output.as_deref())?;
    Ok(receipt.status == "PASS")
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("schema contract audit failed before a verdict: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "$schema": DRAFT_2020_12_URI,
            "type": "object",
            "properties": {
                "scenario_id": { "type": "string", "minLength": 1 },
                "synthetic": { "const": true }
            },
            "required": ["scenario_id", "synthetic"],
            "additionalProperties": false
        })
    }

    #[test]
    fn accepts_valid_draft_2020_12_schema_and_instance() {
        let errors = validate_values(
            &schema(),
            &json!({"scenario_id": "SYN-001", "synthetic": true}),
        )
        .expect("offline schema validation");
        assert!(errors.is_empty());
    }

    #[test]
    fn rejects_instance_with_missing_required_field() {
        let errors = validate_values(&schema(), &json!({"synthetic": true}))
            .expect("schema itself remains valid");
        assert!(!errors.is_empty());
        assert!(errors[0].contains("scenario_id"));
    }

    #[test]
    fn rejects_instance_that_violates_synthetic_only_const() {
        let errors = validate_values(
            &schema(),
            &json!({"scenario_id": "SYN-001", "synthetic": false}),
        )
        .expect("schema itself remains valid");
        assert!(!errors.is_empty());
    }

    #[test]
    fn rejects_a_non_2020_12_dialect() {
        let mut candidate = schema();
        candidate["$schema"] = json!("http://json-schema.org/draft-07/schema#");
        assert!(validate_values(&candidate, &json!({})).is_err());
    }

    #[test]
    fn refuses_unresolved_remote_references_without_fetching() {
        let candidate = json!({
            "$schema": DRAFT_2020_12_URI,
            "$ref": "https://example.invalid/remote-schema.json"
        });
        let result = validate_values(&candidate, &json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("offline Draft 2020-12 validator"));
    }

    #[test]
    fn permits_local_definition_references() {
        let candidate = json!({
            "$schema": DRAFT_2020_12_URI,
            "$defs": { "identifier": { "type": "string", "minLength": 1 } },
            "$ref": "#/$defs/identifier"
        });
        assert!(validate_values(&candidate, &json!("ok"))
            .expect("local references are self-contained")
            .is_empty());
        assert!(!validate_values(&candidate, &json!(""))
            .expect("local references are self-contained")
            .is_empty());
    }

    #[test]
    fn receipt_binds_exact_head_and_raw_input_bytes() {
        let args = Args {
            schema: PathBuf::from("schema.json"),
            instance: PathBuf::from("instance.json"),
            expected_head: "a".repeat(40),
            output: None,
        };
        let schema_bytes = br#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"boolean"}"#;
        let instance_bytes = b"true";
        let receipt = make_receipt(&args, schema_bytes, instance_bytes).unwrap();
        assert_eq!(receipt.status, "PASS");
        assert_eq!(receipt.expected_head, "a".repeat(40));
        assert_eq!(receipt.schema_sha256, sha256(schema_bytes));
        assert_eq!(receipt.instance_sha256, sha256(instance_bytes));
        assert_eq!(receipt.instance_valid, Some(true));
    }

    #[test]
    fn invalid_instance_produces_fail_receipt_with_digests() {
        let args = Args {
            schema: PathBuf::from("schema.json"),
            instance: PathBuf::from("instance.json"),
            expected_head: "b".repeat(40),
            output: None,
        };
        let schema_bytes = br#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"integer","minimum":5}"#;
        let instance_bytes = b"3";
        let receipt = make_receipt(&args, schema_bytes, instance_bytes).unwrap();
        assert_eq!(receipt.status, "FAIL");
        assert_eq!(receipt.instance_valid, Some(false));
        assert!(!receipt.validation_errors.is_empty());
    }
}
