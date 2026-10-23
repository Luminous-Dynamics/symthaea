use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const SUITE_PATH: &str = "docs/security/mesh-security-scenarios-v1.json";
const SUITE_SCHEMA_PATH: &str = "docs/security/mesh-security-scenarios-v1.schema.json";
const RECEIPT_SCHEMA_PATH: &str = "docs/security/mesh-security-evidence-v1.schema.json";
const AUDIT_RECEIPT_SCHEMA_PATH: &str = "docs/security/mesh-security-contract-audit-v1.schema.json";
const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

fn main() {
    let (root, output) = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            emit_failure(&error, None);
            std::process::exit(2);
        }
    };

    match audit(&root) {
        Ok(receipt) => {
            let failed = receipt["status"] != "PASS";
            let emitted = emit(&receipt, output.as_deref());
            if failed || !emitted {
                std::process::exit(1);
            }
        }
        Err(error) => {
            emit_failure(&error, output.as_deref());
            std::process::exit(1);
        }
    }
}

fn parse_args() -> Result<(PathBuf, Option<PathBuf>), String> {
    let mut root = PathBuf::from(".");
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                let value = args.next().ok_or("--root requires a directory")?;
                root = PathBuf::from(value);
            }
            "--out" => {
                let value = args.next().ok_or("--out requires a file path")?;
                output = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!(
                    "mesh-security-contract-audit [--root REPOSITORY_ROOT] [--out RECEIPT.json]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve repository root: {error}"))?;
    Ok((root, output))
}

fn audit(root: &Path) -> Result<Value, String> {
    let suite_path = root.join(SUITE_PATH);
    let suite_schema_path = root.join(SUITE_SCHEMA_PATH);
    let receipt_schema_path = root.join(RECEIPT_SCHEMA_PATH);
    let audit_receipt_schema_path = root.join(AUDIT_RECEIPT_SCHEMA_PATH);

    let suite = read_json(&suite_path)?;
    let suite_schema = read_json(&suite_schema_path)?;
    let receipt_schema = read_json(&receipt_schema_path)?;
    let audit_receipt_schema = read_json(&audit_receipt_schema_path)?;

    let mut errors = Vec::new();

    validate_schema_document(&suite_schema, SUITE_SCHEMA_PATH, &mut errors);
    validate_schema_document(&receipt_schema, RECEIPT_SCHEMA_PATH, &mut errors);
    validate_schema_document(
        &audit_receipt_schema,
        AUDIT_RECEIPT_SCHEMA_PATH,
        &mut errors,
    );

    let suite_instance_status = match validate_instance(&suite_schema, &suite, SUITE_PATH) {
        Ok(()) => "PASS",
        Err(error) => {
            errors.push(error);
            "FAIL"
        }
    };

    let semantics = semantic_checks(&suite);
    for error in semantics["errors"].as_array().into_iter().flatten() {
        errors.push(
            error
                .as_str()
                .unwrap_or("unknown semantic error")
                .to_owned(),
        );
    }

    let source_revision = git_head(root)?;
    let tool_manifest = root.join("tools/mesh-security-contract-audit/Cargo.toml");
    let tool_source = root.join("tools/mesh-security-contract-audit/src/main.rs");
    let lock_path = root.join("tools/mesh-security-contract-audit/Cargo.lock");

    let schema_checks_pass = !errors
        .iter()
        .any(|message| message.starts_with("schema ") || message.starts_with("instance "));
    let semantics_pass = semantics["errors"]
        .as_array()
        .is_some_and(|values| values.is_empty());
    let status = if suite_instance_status == "PASS" && schema_checks_pass && semantics_pass {
        "PASS"
    } else {
        "FAIL"
    };

    let receipt = json!({
        "schema_version": "symthaea.mesh-security-contract-audit/v1",
        "status": status,
        "source_revision": source_revision,
        "suite_id": suite.get("suite_id").cloned().unwrap_or(Value::Null),
        "claim_ceiling": "schema_and_registry_integrity_only",
        "evidence_class": "contract_validation",
        "artifacts": {
            "scenario_suite": {
                "path": SUITE_PATH,
                "sha256": sha256_file(&suite_path)?
            },
            "scenario_schema": {
                "path": SUITE_SCHEMA_PATH,
                "sha256": sha256_file(&suite_schema_path)?,
                "meta_schema_validation": schema_document_status(&suite_schema)
            },
            "evidence_receipt_schema": {
                "path": RECEIPT_SCHEMA_PATH,
                "sha256": sha256_file(&receipt_schema_path)?,
                "meta_schema_validation": schema_document_status(&receipt_schema)
            },
            "audit_receipt_schema": {
                "path": AUDIT_RECEIPT_SCHEMA_PATH,
                "sha256": sha256_file(&audit_receipt_schema_path)?,
                "meta_schema_validation": schema_document_status(&audit_receipt_schema)
            },
            "validator_manifest": {
                "path": "tools/mesh-security-contract-audit/Cargo.toml",
                "sha256": sha256_file(&tool_manifest)?
            },
            "validator_source": {
                "path": "tools/mesh-security-contract-audit/src/main.rs",
                "sha256": sha256_file(&tool_source)?
            },
            "validator_lockfile": {
                "path": "tools/mesh-security-contract-audit/Cargo.lock",
                "sha256": if lock_path.exists() {
                    Value::String(sha256_file(&lock_path)?)
                } else {
                    Value::Null
                }
            }
        },
        "validator": {
            "crate": "jsonschema",
            "version": "0.59.0",
            "external_reference_resolution": "offline",
            "format_validation": true,
            "default_features": false
        },
        "counts": {
            "properties": suite.get("properties").and_then(Value::as_array).map_or(0, Vec::len),
            "scenarios": suite.get("scenarios").and_then(Value::as_array).map_or(0, Vec::len),
            "proposed_scenarios": suite.get("scenarios").and_then(Value::as_array)
                .map_or(0, |items| items.iter().filter(|item| item["status"] == "proposed").count())
        },
        "checks": {
            "scenario_schema_meta_schema": schema_document_status(&suite_schema),
            "receipt_schema_meta_schema": schema_document_status(&receipt_schema),
            "audit_receipt_schema_meta_schema": schema_document_status(&audit_receipt_schema),
            "scenario_instance_validation": suite_instance_status,
            "audit_receipt_instance_validation": "PASS",
            "semantic_consistency": if semantics_pass { "PASS" } else { "FAIL" },
            "unique_property_ids": semantics["unique_property_ids"],
            "unique_scenario_ids": semantics["unique_scenario_ids"],
            "scenario_property_references_resolve": semantics["scenario_property_references_resolve"],
            "source_revision_recorded": true
        },
        "errors": errors,
        "limitations": [
            "This receipt validates JSON schema shape and internal registry consistency only.",
            "It does not execute Rust cryptographic tests or either TLA+ model.",
            "It does not show that any scenario has an executable oracle or that a counterexample occurred.",
            "It does not prove the protocol, primitive, deployment configuration, or implementation secure."
        ]
    });

    match validate_instance(
        &audit_receipt_schema,
        &receipt,
        "generated contract audit receipt",
    ) {
        Ok(()) => Ok(receipt),
        Err(error) => {
            errors.push(error);
            let mut failed_receipt = receipt;
            failed_receipt["status"] = Value::String("FAIL".to_owned());
            failed_receipt["checks"]["audit_receipt_instance_validation"] =
                Value::String("FAIL".to_owned());
            failed_receipt["errors"] = json!(errors);
            Ok(failed_receipt)
        }
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid JSON in {}: {error}", path.display()))
}

fn validate_schema_document(schema: &Value, path: &str, errors: &mut Vec<String>) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        errors.push(format!(
            "schema {path} must explicitly declare Draft 2020-12 ({DRAFT_2020_12})"
        ));
        return;
    }

    match jsonschema::meta::validator_for(schema) {
        Ok(meta_validator) => {
            for error in meta_validator.iter_errors(schema) {
                errors.push(format!(
                    "schema {path} fails Draft 2020-12 meta-schema at {}: {error}",
                    error.instance_path()
                ));
            }
        }
        Err(error) => errors.push(format!(
            "schema {path} cannot be checked against its declared meta-schema: {error}"
        )),
    }
}

fn schema_document_status(schema: &Value) -> &'static str {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        return "FAIL";
    }

    jsonschema::meta::validator_for(schema)
        .ok()
        .is_some_and(|validator| validator.is_valid(schema))
        .then_some("PASS")
        .unwrap_or("FAIL")
}

fn validate_instance(schema: &Value, instance: &Value, label: &str) -> Result<(), String> {
    let validator = jsonschema::options()
        .offline()
        .should_validate_formats(true)
        .build(schema)
        .map_err(|error| format!("cannot build offline validator for {label}: {error}"))?;

    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "instance {label} failed schema validation: {}",
            errors.join("; ")
        ))
    }
}

fn semantic_checks(suite: &Value) -> Value {
    let mut errors = Vec::new();
    let mut property_ids = BTreeSet::new();
    let mut scenario_ids = BTreeSet::new();

    let properties = suite.get("properties").and_then(Value::as_array);
    let scenarios = suite.get("scenarios").and_then(Value::as_array);

    if let Some(properties) = properties {
        for (index, property) in properties.iter().enumerate() {
            match property.get("id").and_then(Value::as_str) {
                Some(id) if property_ids.insert(id.to_owned()) => {}
                Some(id) => errors.push(format!(
                    "duplicate property ID at properties[{index}]: {id}"
                )),
                None => errors.push(format!("properties[{index}] has no string ID")),
            }
        }
    } else {
        errors.push("properties must be an array".to_owned());
    }

    let mut references_resolve = true;
    if let Some(scenarios) = scenarios {
        for (index, scenario) in scenarios.iter().enumerate() {
            match scenario.get("id").and_then(Value::as_str) {
                Some(id) if scenario_ids.insert(id.to_owned()) => {}
                Some(id) => {
                    errors.push(format!("duplicate scenario ID at scenarios[{index}]: {id}"))
                }
                None => errors.push(format!("scenarios[{index}] has no string ID")),
            }

            match scenario.get("property").and_then(Value::as_str) {
                Some(property) if property_ids.contains(property) => {}
                Some(property) => {
                    references_resolve = false;
                    errors.push(format!(
                        "scenario {} references undeclared property {property}",
                        scenario
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("<missing-id>")
                    ));
                }
                None => {
                    references_resolve = false;
                    errors.push(format!(
                        "scenarios[{index}] has no string property reference"
                    ));
                }
            }
        }
    } else {
        errors.push("scenarios must be an array".to_owned());
    }

    json!({
        "unique_property_ids": errors
            .iter()
            .all(|message| !message.starts_with("duplicate property ID")),
        "unique_scenario_ids": errors
            .iter()
            .all(|message| !message.starts_with("duplicate scenario ID")),
        "scenario_property_references_resolve": references_resolve,
        "errors": errors
    })
}

fn git_head(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot execute git rev-parse: {error}"))?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed; exact source revision is required".to_owned());
    }

    let revision = String::from_utf8(output.stdout)
        .map_err(|error| format!("git revision is not UTF-8: {error}"))?
        .trim()
        .to_owned();

    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("git rev-parse HEAD did not return a full 40-character commit SHA".to_owned());
    }
    Ok(revision)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn emit(receipt: &Value, output: Option<&Path>) -> bool {
    let serialized = match serde_json::to_string_pretty(receipt) {
        Ok(serialized) => serialized,
        Err(error) => {
            eprintln!("cannot serialize contract audit receipt: {error}");
            return false;
        }
    };

    if let Some(path) = output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if let Err(error) = fs::create_dir_all(parent) {
                eprintln!(
                    "cannot create receipt directory {}: {error}",
                    parent.display()
                );
                println!("{serialized}");
                return false;
            }
        }
        if let Err(error) = fs::write(path, &serialized) {
            eprintln!("cannot write receipt {}: {error}", path.display());
            println!("{serialized}");
            return false;
        }
    }

    println!("{serialized}");
    true
}

fn emit_failure(error: &str, output: Option<&Path>) {
    let receipt = json!({
        "schema_version": "symthaea.mesh-security-contract-audit/v1",
        "status": "FAIL",
        "claim_ceiling": "none",
        "errors": [error],
        "limitations": [
            "Audit did not complete; no schema or registry pass is implied."
        ]
    });
    let _ = emit(&receipt, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root must resolve")
    }

    #[test]
    fn checked_in_contracts_validate_and_bind_revision() {
        let receipt = audit(&repository_root()).expect("audit should complete");
        assert_eq!(receipt["status"], "PASS", "{receipt}");
        assert_eq!(receipt["source_revision"].as_str().unwrap().len(), 40);
        assert_eq!(receipt["checks"]["scenario_schema_meta_schema"], "PASS");
        assert_eq!(receipt["checks"]["receipt_schema_meta_schema"], "PASS");
        assert_eq!(receipt["checks"]["scenario_instance_validation"], "PASS");
    }

    #[test]
    fn duplicate_property_and_scenario_ids_are_rejected() {
        let mut suite: Value = serde_json::from_str(include_str!(
            "../../../docs/security/mesh-security-scenarios-v1.json"
        ))
        .expect("checked-in suite parses");
        let property = suite["properties"][0].clone();
        suite["properties"].as_array_mut().unwrap().push(property);
        let scenario = suite["scenarios"][0].clone();
        suite["scenarios"].as_array_mut().unwrap().push(scenario);

        let errors = semantic_checks(&suite)["errors"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            errors
                .iter()
                .any(|error| error.as_str().unwrap().contains("duplicate property ID"))
        );
        assert!(
            errors
                .iter()
                .any(|error| error.as_str().unwrap().contains("duplicate scenario ID"))
        );
    }

    #[test]
    fn undeclared_scenario_property_is_rejected() {
        let suite: Value = serde_json::from_str(include_str!(
            "../../../docs/security/mesh-security-scenarios-v1.json"
        ))
        .expect("checked-in suite parses");
        let mut mutated = suite;
        mutated["scenarios"][0]["property"] = json!("UNDECLARED-SECURITY-PROPERTY");

        let errors = semantic_checks(&mutated)["errors"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            errors
                .iter()
                .any(|error| error.as_str().unwrap().contains("undeclared property"))
        );
    }

    #[test]
    fn schema_auditor_rejects_external_references_offline() {
        let external_schema = json!({
            "$schema": DRAFT_2020_12,
            "$ref": "https://schemas.example.invalid/mesh.json"
        });
        let error = validate_instance(
            &external_schema,
            &json!({"attack": true}),
            "offline negative control",
        )
        .expect_err("unregistered external schema must not be fetched");
        assert!(error.contains("offline validator"), "{error}");
    }

    #[test]
    fn schema_auditor_rejects_invalid_schema_keyword_types() {
        let invalid_schema = json!({
            "$schema": DRAFT_2020_12,
            "type": 42
        });
        let mut errors = Vec::new();
        validate_schema_document(&invalid_schema, "invalid-fixture.schema.json", &mut errors);
        assert!(
            !errors.is_empty(),
            "invalid JSON Schema must fail its meta-schema"
        );
    }

    #[test]
    fn receipt_write_failure_is_not_reported_as_success() {
        let blocker = env::temp_dir().join(format!(
            "mesh-security-audit-output-blocker-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock follows Unix epoch")
                .as_nanos()
        ));
        fs::write(&blocker, b"not a directory").expect("create a file to block directory creation");
        let output = blocker.join("receipt.json");

        assert!(
            !emit(&json!({"status": "PASS"}), Some(&output)),
            "a passing audit whose receipt cannot be written must have a non-success emission result"
        );

        fs::remove_file(blocker).expect("remove output blocker");
    }
}
