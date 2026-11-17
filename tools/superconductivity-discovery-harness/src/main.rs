use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn reject_schema_references(value: &Value, path: &str) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            for key in ["$ref", "$dynamicRef", "$recursiveRef"] {
                if object.contains_key(key) {
                    return Err(format!(
                        "schema contains forbidden external or indirect reference {key} at {path}"
                    ));
                }
            }
            for (key, child) in object {
                reject_schema_references(child, &format!("{path}/{key}"))?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_schema_references(child, &format!("{path}/{index}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn required_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{context}: missing or empty string field {key}"))
}

fn required_array<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{context}: missing array field {key}"))
}

fn audit_registry(schema_bytes: &[u8], registry_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let registry: Value = serde_json::from_slice(registry_bytes)
        .map_err(|error| format!("registry JSON parse failed: {error}"))?;

    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!(
            "schema must declare Draft 2020-12 exactly ({DIALECT})"
        ));
    }
    reject_schema_references(&schema, "#")?;

    // The dependency is built with default features disabled, so the validator
    // cannot retrieve schemas from HTTP or the filesystem. The explicit
    // reference scan above also keeps this registry contract self-contained.
    let validator = validator_for(&schema)
        .map_err(|error| format!("schema compilation / meta-validation failed: {error}"))?;
    let schema_errors: Vec<String> = validator
        .iter_errors(&registry)
        .map(|error| format!("{} at {}", error, error.instance_path()))
        .collect();
    if !schema_errors.is_empty() {
        return Err(format!(
            "registry failed JSON Schema validation:\n{}",
            schema_errors.join("\n")
        ));
    }

    let sources = required_array(&registry, "source_register", "registry")?;
    let candidates = required_array(&registry, "candidate_records", "registry")?;
    let catalogs = required_array(&registry, "benchmark_catalogs", "registry")?;

    let mut source_ids = HashSet::with_capacity(sources.len());
    for source in sources {
        let id = required_str(source, "source_id", "source record")?;
        if !source_ids.insert(id.to_owned()) {
            return Err(format!("duplicate source_id: {id}"));
        }
    }

    let mut candidate_ids = HashSet::with_capacity(candidates.len());
    let mut candidate_id_list = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let context = "candidate record";
        let candidate_id = required_str(candidate, "candidate_id", context)?;
        if !candidate_ids.insert(candidate_id.to_owned()) {
            return Err(format!("duplicate candidate_id: {candidate_id}"));
        }
        candidate_id_list.push(candidate_id.to_owned());

        let references = required_array(candidate, "source_ids", context)?;
        if references.is_empty() {
            return Err(format!("{candidate_id}: source_ids must not be empty"));
        }
        for reference in references {
            let reference = reference
                .as_str()
                .ok_or_else(|| format!("{candidate_id}: source_ids must contain strings"))?;
            if !source_ids.contains(reference) {
                return Err(format!("{candidate_id}: unresolved source_id {reference}"));
            }
        }

        let evidence_class = required_str(candidate, "evidence_class", context)?;
        let observations = required_array(candidate, "tc_observations", context)?;
        let predictions = match candidate.get("tc_prediction") {
            None | Some(Value::Null) => &[][..],
            Some(Value::Array(items)) => items.as_slice(),
            Some(_) => {
                return Err(format!(
                    "{candidate_id}: tc_prediction must be array or null"
                ));
            }
        };
        let calculations = match candidate.get("tc_calculations") {
            None | Some(Value::Null) => &[][..],
            Some(Value::Array(items)) => items.as_slice(),
            Some(_) => {
                return Err(format!(
                    "{candidate_id}: tc_calculations must be array or null"
                ));
            }
        };

        for observation in observations {
            let status = required_str(observation, "status", candidate_id)?;
            if status.starts_with("predicted") || status.starts_with("calculated") {
                return Err(format!(
                    "{candidate_id}: non-observed Tc status {status} is stored in tc_observations"
                ));
            }
        }
        for prediction in predictions {
            let status = required_str(prediction, "status", candidate_id)?;
            if !status.starts_with("predicted") {
                return Err(format!(
                    "{candidate_id}: tc_prediction status must start with 'predicted', got {status}"
                ));
            }
        }
        for calculation in calculations {
            let status = required_str(calculation, "status", candidate_id)?;
            if !status.starts_with("calculated") {
                return Err(format!(
                    "{candidate_id}: tc_calculations status must start with 'calculated', got {status}"
                ));
            }
        }

        match evidence_class {
            "computational_prediction" | "machine_learning_prediction" => {
                if !observations.is_empty() {
                    return Err(format!(
                        "{candidate_id}: predicted evidence must not be stored as observed Tc"
                    ));
                }
                if predictions.is_empty() || !calculations.is_empty() {
                    return Err(format!(
                        "{candidate_id}: prediction evidence class requires tc_prediction entries and no tc_calculations"
                    ));
                }
            }
            "experimental_report" => {
                if observations.is_empty() {
                    return Err(format!(
                        "{candidate_id}: experimental report requires at least one observed transition"
                    ));
                }
                // Experimental materials may also have separate, explicitly
                // calculated or predicted Tc estimates; these stay in their
                // own fields and never overwrite measured observations.
            }
            _ => {}
        }

        let formula = required_str(candidate, "formula", candidate_id)?;
        if formula.contains("Pu") || formula.contains("Ac") {
            let safety = required_str(candidate, "safety_review", candidate_id)?;
            if !safety.contains("computational_only") && !safety.contains("exclude") {
                return Err(format!(
                    "{candidate_id}: actinide-bearing candidate lacks computational-only / exclusion gate"
                ));
            }
        }
    }

    for catalog in catalogs {
        let catalog_id = required_str(catalog, "catalog_id", "benchmark catalog")?;
        let references = required_array(catalog, "source_ids", catalog_id)?;
        for reference in references {
            let reference = reference
                .as_str()
                .ok_or_else(|| format!("{catalog_id}: source_ids must contain strings"))?;
            if !source_ids.contains(reference) {
                return Err(format!("{catalog_id}: unresolved source_id {reference}"));
            }
        }
    }

    Ok(json!({
        "status": "pass",
        "validator": "jsonschema 0.59.0 + registry semantic audit",
        "schema_dialect": DIALECT,
        "schema_sha256": sha256_hex(schema_bytes),
        "registry_sha256": sha256_hex(registry_bytes),
        "source_count": sources.len(),
        "candidate_count": candidates.len(),
        "benchmark_catalog_count": catalogs.len(),
        "candidate_ids": candidate_id_list,
        "checks": {
            "schema_meta_validation_and_instance_validation": true,
            "schema_has_no_external_or_indirect_references": true,
            "source_ids_unique": true,
            "candidate_ids_unique": true,
            "all_source_references_resolve": true,
            "prediction_and_observation_channels_separated": true,
            "actinide_candidate_experiment_gate_present": true
        }
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --instance <registry.json> [--out <receipt.json>]"
    );
}

fn main() {
    let mut schema_path: Option<PathBuf> = None;
    let mut instance_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let args: Vec<String> = env::args().collect();
    let mut index = 1;

    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--instance" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let value = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(value),
                    "--instance" => instance_path = Some(value),
                    "--out" => output_path = Some(value),
                    _ => unreachable!(),
                }
                index += 2;
            }
            "--help" | "-h" => {
                usage(&args[0]);
                return;
            }
            unknown => {
                eprintln!("unknown argument: {unknown}");
                usage(&args[0]);
                process::exit(2);
            }
        }
    }

    let (Some(schema_path), Some(instance_path)) = (schema_path, instance_path) else {
        usage(&args[0]);
        process::exit(2);
    };

    let schema_bytes = read_file(&schema_path);
    let registry_bytes = read_file(&instance_path);
    let receipt = match audit_registry(&schema_bytes, &registry_bytes) {
        Ok(receipt) => receipt,
        Err(error) => {
            eprintln!("SC-DISCOVERY-REGISTRY-FAIL: {error}");
            process::exit(1);
        }
    };
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing the audit receipt should not fail");

    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write receipt {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

fn read_file(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn committed_inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema =
            fs::read(root.join("data/schemas/superconductivity_discovery_case_v0_1.schema.json"))
                .expect("committed schema should be readable");
        let registry =
            fs::read(root.join("data/research/sc_superconductor_discovery_cases_v0_1.json"))
                .expect("committed registry should be readable");
        (schema, registry)
    }

    #[test]
    fn committed_registry_passes_schema_and_semantic_audit() {
        let (schema, registry) = committed_inputs();
        let receipt = audit_registry(&schema, &registry).expect("registry must pass");
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["source_count"], 12);
        assert_eq!(receipt["candidate_count"], 8);
        assert_eq!(receipt["benchmark_catalog_count"], 8);
    }

    #[test]
    fn duplicate_candidate_ids_fail_closed() {
        let (schema, registry) = committed_inputs();
        let mut value: Value = serde_json::from_slice(&registry).unwrap();
        let records = value["candidate_records"].as_array_mut().unwrap();
        let duplicate = records[0].clone();
        records.push(duplicate);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(audit_registry(&schema, &mutated).is_err());
    }

    #[test]
    fn orphan_source_references_fail_closed() {
        let (schema, registry) = committed_inputs();
        let mut value: Value = serde_json::from_slice(&registry).unwrap();
        value["candidate_records"][0]["source_ids"] = json!(["missing-source"]);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(audit_registry(&schema, &mutated).is_err());
    }

    #[test]
    fn predictions_cannot_be_promoted_to_observed_transitions() {
        let (schema, registry) = committed_inputs();
        let mut value: Value = serde_json::from_slice(&registry).unwrap();
        value["candidate_records"][2]["tc_observations"] = json!([{
            "observable": "zero_resistance",
            "value_k": 44.0,
            "status": "reported"
        }]);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(audit_registry(&schema, &mutated).is_err());
    }

    #[test]
    fn experimental_observation_and_calculation_channels_coexist() {
        let (schema, registry) = committed_inputs();
        let value: Value = serde_json::from_slice(&registry).unwrap();
        let candidate = &value["candidate_records"][5];
        assert_eq!(candidate["tc_observations"][0]["value_k"], 3.18);
        assert_eq!(candidate["tc_calculations"][0]["value_k"], 14.9);
        let receipt = audit_registry(&schema, &registry).unwrap();
        assert_eq!(receipt["status"], "pass");
    }

    #[test]
    fn calculated_value_cannot_be_stored_as_an_observation() {
        let (schema, registry) = committed_inputs();
        let mut value: Value = serde_json::from_slice(&registry).unwrap();
        value["candidate_records"][5]["tc_observations"][0]["status"] = json!("calculated");
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(audit_registry(&schema, &mutated).is_err());
    }

    #[test]
    fn calculation_status_cannot_be_mislabeled_as_prediction() {
        let (schema, registry) = committed_inputs();
        let mut value: Value = serde_json::from_slice(&registry).unwrap();
        value["candidate_records"][5]["tc_prediction"] = json!([{
            "value_k": 14.9,
            "pressure_gpa": 0.0,
            "method": "DFT_Eliashberg",
            "status": "calculated"
        }]);
        let mutated = serde_json::to_vec(&value).unwrap();
        assert!(audit_registry(&schema, &mutated).is_err());
    }

    #[test]
    fn external_schema_references_are_refused() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "https://example.invalid/remote.json",
            "type": "object"
        });
        assert!(reject_schema_references(&schema, "#").is_err());
    }
}
