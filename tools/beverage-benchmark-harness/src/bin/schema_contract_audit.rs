//! Offline Draft 2020-12 schema and synthetic Beverage benchmark auditor.
//!
//! The executable validates the schema document against the bundled Draft 2020-12
//! meta-schema, validates the corpus instance, and applies a small set of explicit
//! fixture-only invariants. Remote and file reference retrieval is disabled.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const DRAFT_2020_12_URI: &str = "https://json-schema.org/draft/2020-12/schema";
const BEVERAGE_BENCHMARK: &str = "BEVERAGE_ENGINEERING_BENCHMARK_V1";

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
    semantic_checks_passed: Option<bool>,
    validation_errors: Vec<String>,
    semantic_errors: Vec<String>,
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

fn validate_schema_instance(schema: &Value, instance: &Value) -> Result<Vec<String>, String> {
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

fn expected_fixture_disposition(scenario: &Value) -> Option<&'static str> {
    let string = |key: &str| scenario.get(key).and_then(Value::as_str);
    let asserted_as_measured = scenario
        .get("asserted_as_measured")
        .and_then(Value::as_bool)?;

    let evidence_class = string("evidence_class")?;
    let data_integrity = string("data_integrity")?;
    let rights_status = string("rights_status")?;
    let calibration_status = string("calibration_status")?;
    let split_status = string("split_status")?;
    let cutoff_status = string("cutoff_status")?;
    let domain_shift = string("domain_shift")?;
    let hard_gate = string("hard_gate")?;

    let reject = matches!(
        data_integrity,
        "group_leakage" | "post_cutoff_feature" | "unit_inconsistent"
    ) || (asserted_as_measured && evidence_class != "MEASURED")
        || hard_gate == "violated";
    if reject {
        return Some("REJECT");
    }

    let indeterminate = evidence_class == "UNKNOWN"
        || rights_status != "verified"
        || matches!(calibration_status, "missing" | "expired" | "unknown")
        || split_status != "batch_grouped"
        || cutoff_status == "unknown"
        || hard_gate == "unknown"
        || domain_shift == "unknown";
    if indeterminate {
        return Some("INDETERMINATE");
    }

    if domain_shift == "known_shift" {
        Some("ELIGIBLE_WITH_SHIFT_WARNING")
    } else {
        Some("ELIGIBLE")
    }
}

fn validate_beverage_fixture_semantics(instance: &Value) -> Option<Vec<String>> {
    if instance.get("benchmark").and_then(Value::as_str) != Some(BEVERAGE_BENCHMARK) {
        return None;
    }

    let mut errors = Vec::new();
    if instance.get("fixture_class").and_then(Value::as_str)
        != Some("SYNTHETIC_CONTRACT_FIXTURES_ONLY")
    {
        errors.push("fixture_class must remain SYNTHETIC_CONTRACT_FIXTURES_ONLY".to_owned());
    }

    let Some(scenarios) = instance.get("scenarios").and_then(Value::as_array) else {
        errors.push("scenarios must be an array".to_owned());
        return Some(errors);
    };

    let mut identifiers = HashSet::new();
    for (index, scenario) in scenarios.iter().enumerate() {
        let location = format!("scenarios[{index}]");
        let Some(identifier) = scenario.get("scenario_id").and_then(Value::as_str) else {
            errors.push(format!("{location}: scenario_id must be a string"));
            continue;
        };
        if !identifiers.insert(identifier.to_owned()) {
            errors.push(format!("{location}: duplicate scenario_id {identifier:?}"));
        }
        if scenario.get("synthetic").and_then(Value::as_bool) != Some(true) {
            errors.push(format!("{location}: every fixture must declare synthetic=true"));
        }

        let expected = scenario
            .get("expected_disposition")
            .and_then(Value::as_str);
        match expected_fixture_disposition(scenario) {
            Some(calculated) if expected == Some(calculated) => {}
            Some(calculated) => errors.push(format!(
                "{location}: expected_disposition {:?} does not match fixture oracle {calculated}",
                expected
            )),
            None => errors.push(format!(
                "{location}: missing or malformed input for the deterministic fixture oracle"
            )),
        }
    }

    Some(errors)
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
        match validate_schema_instance(&schema, &instance) {
            Ok(errors) => (true, true, Some(errors.is_empty()), errors),
            Err(error) => {
                let meta_valid = jsonschema::draft202012::meta::validate(&schema).is_ok()
                    && schema_uri == DRAFT_2020_12_URI;
                (meta_valid, false, None, vec![error])
            }
        };

    let semantic_errors = if instance_valid == Some(true) {
        validate_beverage_fixture_semantics(&instance).unwrap_or_default()
    } else {
        Vec::new()
    };
    let semantic_checks_passed = if instance_valid == Some(true)
        && instance.get("benchmark").and_then(Value::as_str) == Some(BEVERAGE_BENCHMARK)
    {
        Some(semantic_errors.is_empty())
    } else {
        None
    };

    let status = if schema_meta_valid
        && validator_built
        && instance_valid == Some(true)
        && semantic_checks_passed.is_none_or(|passed| passed)
    {
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
        semantic_checks_passed,
        validation_errors,
        semantic_errors,
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

    fn scenario(id: &str) -> Value {
        json!({
            "scenario_id": id,
            "synthetic": true,
            "evidence_class": "MEASURED",
            "asserted_as_measured": true,
            "data_integrity": "valid",
            "rights_status": "verified",
            "calibration_status": "verified",
            "split_status": "batch_grouped",
            "cutoff_status": "pre_cutoff",
            "domain_shift": "none",
            "hard_gate": "clear",
            "expected_disposition": "ELIGIBLE"
        })
    }

    fn corpus(scenarios: Vec<Value>) -> Value {
        json!({
            "schema_version": "1.0.0",
            "benchmark": BEVERAGE_BENCHMARK,
            "fixture_class": "SYNTHETIC_CONTRACT_FIXTURES_ONLY",
            "scenarios": scenarios
        })
    }

    fn schema() -> Value {
        json!({
            "$schema": DRAFT_2020_12_URI,
            "type": "object",
            "required": ["schema_version", "benchmark", "fixture_class", "scenarios"],
            "properties": {
                "schema_version": { "const": "1.0.0" },
                "benchmark": { "const": BEVERAGE_BENCHMARK },
                "fixture_class": { "const": "SYNTHETIC_CONTRACT_FIXTURES_ONLY" },
                "scenarios": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "scenario_id", "synthetic", "evidence_class",
                            "asserted_as_measured", "data_integrity", "rights_status",
                            "calibration_status", "split_status", "cutoff_status",
                            "domain_shift", "hard_gate", "expected_disposition"
                        ],
                        "properties": {
                            "scenario_id": { "type": "string", "minLength": 1 },
                            "synthetic": { "const": true },
                            "evidence_class": { "enum": ["MEASURED", "MODELLED", "SIMULATED", "DECLARED", "UNKNOWN"] },
                            "asserted_as_measured": { "type": "boolean" },
                            "data_integrity": { "enum": ["valid", "group_leakage", "post_cutoff_feature", "unit_inconsistent"] },
                            "rights_status": { "enum": ["verified", "unknown", "restricted"] },
                            "calibration_status": { "enum": ["verified", "missing", "expired", "unknown"] },
                            "split_status": { "enum": ["batch_grouped", "row_randomized", "unknown"] },
                            "cutoff_status": { "enum": ["pre_cutoff", "post_cutoff_feature", "unknown", "not_applicable"] },
                            "domain_shift": { "enum": ["none", "known_shift", "unknown"] },
                            "hard_gate": { "enum": ["clear", "violated", "unknown"] },
                            "expected_disposition": { "enum": ["ELIGIBLE", "REJECT", "INDETERMINATE", "ELIGIBLE_WITH_SHIFT_WARNING"] }
                        },
                        "additionalProperties": false
                    }
                }
            },
            "additionalProperties": false
        })
    }

    #[test]
    fn accepts_valid_draft_2020_12_schema_and_synthetic_corpus() {
        let data = corpus(vec![scenario("BEV-SYN-001")]);
        let errors = validate_schema_instance(&schema(), &data)
            .expect("offline schema validation");
        assert!(errors.is_empty());
        assert!(validate_beverage_fixture_semantics(&data)
            .expect("beverage fixture semantics")
            .is_empty());
    }

    #[test]
    fn rejects_instance_with_missing_required_field() {
        let mut data = corpus(vec![scenario("BEV-SYN-001")]);
        data["scenarios"][0].as_object_mut().unwrap().remove("scenario_id");
        assert!(!validate_schema_instance(&schema(), &data)
            .expect("schema itself remains valid")
            .is_empty());
    }

    #[test]
    fn rejects_a_non_2020_12_dialect() {
        let mut candidate = schema();
        candidate["$schema"] = json!("http://json-schema.org/draft-07/schema#");
        assert!(validate_schema_instance(&candidate, &corpus(vec![scenario("BEV-SYN-001")])).is_err());
    }

    #[test]
    fn refuses_unresolved_remote_references_without_fetching() {
        let candidate = json!({
            "$schema": DRAFT_2020_12_URI,
            "$ref": "https://example.invalid/remote-schema.json"
        });
        let result = validate_schema_instance(&candidate, &json!({}));
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
        assert!(validate_schema_instance(&candidate, &json!("ok"))
            .expect("local references are self-contained")
            .is_empty());
        assert!(!validate_schema_instance(&candidate, &json!(""))
            .expect("local references are self-contained")
            .is_empty());
    }

    #[test]
    fn rejects_duplicate_scenario_ids() {
        let data = corpus(vec![scenario("BEV-SYN-001"), scenario("BEV-SYN-001")]);
        let errors = validate_beverage_fixture_semantics(&data).unwrap();
        assert!(errors.iter().any(|error| error.contains("duplicate scenario_id")));
    }

    #[test]
    fn rejects_fixtures_that_are_not_marked_synthetic() {
        let mut data = corpus(vec![scenario("BEV-SYN-001")]);
        data["scenarios"][0]["synthetic"] = json!(false);
        let errors = validate_beverage_fixture_semantics(&data).unwrap();
        assert!(errors.iter().any(|error| error.contains("synthetic=true")));
    }

    #[test]
    fn oracle_rejects_leakage_and_measurement_class_confusion() {
        let mut leakage = scenario("BEV-SYN-002");
        leakage["data_integrity"] = json!("group_leakage");
        leakage["expected_disposition"] = json!("REJECT");
        assert_eq!(expected_fixture_disposition(&leakage), Some("REJECT"));

        let mut confusion = scenario("BEV-SYN-003");
        confusion["evidence_class"] = json!("MODELLED");
        confusion["expected_disposition"] = json!("REJECT");
        assert_eq!(expected_fixture_disposition(&confusion), Some("REJECT"));
    }

    #[test]
    fn oracle_preserves_shift_warning_and_indeterminate_controls() {
        let mut shifted = scenario("BEV-SYN-004");
        shifted["domain_shift"] = json!("known_shift");
        shifted["expected_disposition"] = json!("ELIGIBLE_WITH_SHIFT_WARNING");
        assert_eq!(
            expected_fixture_disposition(&shifted),
            Some("ELIGIBLE_WITH_SHIFT_WARNING")
        );

        let mut unknown = scenario("BEV-SYN-005");
        unknown["calibration_status"] = json!("missing");
        unknown["expected_disposition"] = json!("INDETERMINATE");
        assert_eq!(expected_fixture_disposition(&unknown), Some("INDETERMINATE"));
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
