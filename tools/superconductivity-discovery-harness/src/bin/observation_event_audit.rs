use jsonschema::validator_for;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::process;

fn validate_fixture(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;

    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator =
        validator_for(&schema).map_err(|error| format!("schema compilation failed: {error}"))?;

    let fixture_sources = fixture
        .get("fixture_source_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "fixture_source_ids must be an array".to_owned())?;
    let mut source_ids = HashSet::new();
    for source in fixture_sources {
        let id = source
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "fixture_source_ids must contain non-empty strings".to_owned())?;
        if !source_ids.insert(id.to_owned()) {
            return Err(format!("duplicate fixture source id: {id}"));
        }
    }

    let events = fixture
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "observations must be an array".to_owned())?;
    if events.is_empty() {
        return Err("observation fixture must not be empty".to_owned());
    }

    let mut observation_ids = HashSet::new();
    let mut origin_counts = serde_json::Map::new();
    let mut checked = Vec::with_capacity(events.len());

    for (index, event) in events.iter().enumerate() {
        let context = format!("observation[{index}]");
        for error in validator.iter_errors(event) {
            return Err(format!(
                "{context}: schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }

        if event.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err(format!(
                "{context}: only synthetic fixture records are admitted here"
            ));
        }

        let observation_id = event
            .get("observation_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("{context}: missing observation_id"))?;
        if !observation_ids.insert(observation_id.to_owned()) {
            return Err(format!("duplicate observation_id: {observation_id}"));
        }

        let source_id = event
            .pointer("/source/source_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{observation_id}: missing source.source_id"))?;
        if !source_ids.contains(source_id) {
            return Err(format!(
                "{observation_id}: source_id {source_id} is not declared by the fixture"
            ));
        }

        let locator = event
            .pointer("/source/locator")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{observation_id}: missing source locator"))?;
        let locator_fields = [
            "page",
            "section",
            "table",
            "figure",
            "supplementary_file",
            "row_or_record_id",
        ];
        if !locator_fields.iter().any(|field| {
            locator.get(*field).is_some_and(|value| match value {
                Value::String(text) => !text.trim().is_empty(),
                Value::Number(number) => number.as_u64().is_some_and(|n| n > 0),
                _ => false,
            })
        }) {
            return Err(format!(
                "{observation_id}: source locator must identify page, section, table, figure, supplement or source row"
            ));
        }

        let assertion = event
            .get("assertion")
            .ok_or_else(|| format!("{observation_id}: missing assertion"))?;
        let origin = assertion
            .get("origin")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{observation_id}: missing assertion.origin"))?;
        let detection = assertion
            .get("detection_status")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{observation_id}: missing detection_status"))?;
        let value = assertion
            .get("value")
            .ok_or_else(|| format!("{observation_id}: missing assertion.value"))?;
        let property = assertion
            .get("property_code")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{observation_id}: missing property_code"))?;
        let criterion = assertion.get("tc_criterion").and_then(Value::as_str);

        match origin {
            "observed" => {
                if detection == "not_applicable" {
                    return Err(format!(
                        "{observation_id}: observed assertion cannot use not_applicable detection"
                    ));
                }
                if detection == "not_detected" {
                    if !value.is_null() {
                        return Err(format!(
                            "{observation_id}: non-detection must have null value, never Tc=0"
                        ));
                    }
                    let window = assertion
                        .get("tested_window")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            format!("{observation_id}: non-detection requires tested_window")
                        })?;
                    let min = window
                        .get("temperature_min_k")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| {
                            format!("{observation_id}: missing tested minimum temperature")
                        })?;
                    let max = window
                        .get("temperature_max_k")
                        .and_then(Value::as_f64)
                        .ok_or_else(|| {
                            format!("{observation_id}: missing tested maximum temperature")
                        })?;
                    if max < min {
                        return Err(format!(
                            "{observation_id}: tested temperature window is inverted"
                        ));
                    }
                } else if detection == "detected" {
                    if !value.is_number() {
                        return Err(format!(
                            "{observation_id}: detected assertion requires numeric value"
                        ));
                    }
                    if assertion.get("tested_window").is_some_and(|v| !v.is_null()) {
                        return Err(format!(
                            "{observation_id}: tested_window is reserved for non-detections"
                        ));
                    }
                }

                if property == "Tc" && criterion.is_none_or(|value| value == "not_applicable") {
                    return Err(format!(
                        "{observation_id}: observed Tc requires a meaningful measurement criterion"
                    ));
                }
                if property == "Tc" && value.as_f64().is_some_and(|measurement| measurement < 0.0) {
                    return Err(format!("{observation_id}: observed Tc cannot be negative"));
                }
            }
            "calculated" | "predicted" => {
                if detection != "not_applicable" {
                    return Err(format!(
                        "{observation_id}: {origin} assertion cannot be labelled as experimentally detected"
                    ));
                }
                if !value.is_number() {
                    return Err(format!(
                        "{observation_id}: {origin} assertion requires a numeric value"
                    ));
                }
                let model_id = assertion
                    .get("model_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                if model_id.is_none() {
                    return Err(format!(
                        "{observation_id}: {origin} assertion requires model_id"
                    ));
                }
                if property == "Tc" && criterion.is_some_and(|c| c != "not_applicable") {
                    return Err(format!(
                        "{observation_id}: calculated/predicted Tc must not claim an experimental criterion"
                    ));
                }
            }
            "inferred" => {
                if detection != "not_applicable" {
                    return Err(format!(
                        "{observation_id}: inferred assertions cannot claim a measured detection"
                    ));
                }
                if !value.is_number() && !value.is_null() {
                    return Err(format!(
                        "{observation_id}: inferred value must be numeric or null"
                    ));
                }
            }
            other => return Err(format!("{observation_id}: unsupported origin {other}")),
        }

        let pressure = event
            .pointer("/conditions/pressure")
            .ok_or_else(|| format!("{observation_id}: missing pressure semantics"))?;
        let pressure_state = pressure
            .get("semantics")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{observation_id}: missing pressure.semantics"))?;
        let pressure_value = pressure
            .get("value_gpa")
            .ok_or_else(|| format!("{observation_id}: missing pressure.value_gpa"))?;
        match pressure_state {
            "explicit_ambient" => {
                if pressure_value.as_f64() != Some(0.0) {
                    return Err(format!(
                        "{observation_id}: explicit_ambient requires pressure value exactly 0 GPa"
                    ));
                }
            }
            "applied" => {
                if !pressure_value.as_f64().is_some_and(|value| value > 0.0) {
                    return Err(format!(
                        "{observation_id}: applied pressure requires value > 0 GPa"
                    ));
                }
            }
            "unknown" | "chemical" => {
                if !pressure_value.is_null() {
                    return Err(format!(
                        "{observation_id}: {pressure_state} pressure must not be coerced to a numeric GPa value"
                    ));
                }
            }
            other => {
                return Err(format!(
                    "{observation_id}: unsupported pressure semantics {other}"
                ));
            }
        }

        let next_count = origin_counts
            .get(origin)
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        origin_counts.insert(origin.to_owned(), json!(next_count));
        checked.push(observation_id.to_owned());
    }

    Ok(json!({
        "status": "pass",
        "contract": "SC-OBSERVATION-EVENT-0.1",
        "schema_dialect": DIALECT,
        "observation_count": events.len(),
        "source_count": source_ids.len(),
        "origin_counts": origin_counts,
        "observation_ids": checked,
        "checks": {
            "schema_valid": true,
            "source_ids_resolve": true,
            "observation_ids_unique": true,
            "non_detection_is_not_zero": true,
            "calculated_and_predicted_are_not_observations": true,
            "pressure_missingness_preserved": true,
            "ambient_pressure_is_explicit": true
        },
        "claim_ceiling": "Fixture validation only; it does not establish scientific truth, independent replication or a material discovery."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <observation-schema.json> --fixture <events.json> [--out <receipt.json>]"
    );
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut schema_path: Option<PathBuf> = None;
    let mut fixture_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
                    "--out" => output_path = Some(path),
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
    let (Some(schema_path), Some(fixture_path)) = (schema_path, fixture_path) else {
        usage(&args[0]);
        process::exit(2);
    };
    let schema = read(&schema_path);
    let fixture = read(&fixture_path);
    let receipt = validate_fixture(&schema, &fixture).unwrap_or_else(|error| {
        eprintln!("SC-OBSERVATION-EVENT-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing the observation audit receipt should not fail");
    if let Some(path) = output_path {
        if let Err(error) = fs::write(&path, format!("{rendered}\n")) {
            eprintln!("could not write {}: {error}", path.display());
            process::exit(1);
        }
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn committed_inputs() -> (Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_observation_event_v0_1.schema.json"),
        )
        .expect("observation schema must be readable");
        let fixture: Value = serde_json::from_slice(
            &fs::read(root.join("tools/superconductivity-discovery-harness/fixtures/observation_events_smoke_v0_1.json"))
                .expect("observation fixture must be readable")
        ).expect("observation fixture must parse");
        (schema, fixture)
    }

    fn audit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = committed_inputs();
        let bytes = serde_json::to_vec(&fixture).unwrap();
        validate_fixture(&schema, &bytes)
    }

    #[test]
    fn committed_observation_fixture_passes() {
        let (schema, fixture) = committed_inputs();
        let bytes = serde_json::to_vec(&fixture).unwrap();
        let receipt = validate_fixture(&schema, &bytes).unwrap();
        assert_eq!(receipt["observation_count"], 4);
        assert_eq!(receipt["origin_counts"]["observed"], 2);
        assert_eq!(receipt["origin_counts"]["calculated"], 1);
        assert_eq!(receipt["origin_counts"]["predicted"], 1);
    }

    #[test]
    fn non_detection_cannot_be_encoded_as_zero_tc() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][1]["assertion"]["value"] = json!(0.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn unknown_pressure_cannot_be_coerced_to_ambient() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][1]["conditions"]["pressure"]["value_gpa"] = json!(0.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn ambient_pressure_requires_explicit_zero() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][0]["conditions"]["pressure"]["semantics"] =
            json!("explicit_ambient");
        fixture["observations"][0]["conditions"]["pressure"]["value_gpa"] = Value::Null;
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn applied_pressure_requires_positive_pressure_value() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][2]["conditions"]["pressure"]["value_gpa"] = json!(0.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn predicted_value_cannot_claim_observed_detection() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][3]["assertion"]["detection_status"] = json!("detected");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn source_locator_is_required_and_must_be_nonempty() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][0]["source"]["locator"] = json!({});
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn observed_tc_cannot_use_not_applicable_criterion() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][0]["assertion"]["tc_criterion"] = json!("not_applicable");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn negative_observed_tc_fails_closed() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][0]["assertion"]["value"] = json!(-1.0);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn source_id_must_resolve_and_observation_ids_must_be_unique() {
        let (_, mut fixture) = committed_inputs();
        fixture["observations"][0]["source"]["source_id"] = json!("missing-source");
        assert!(audit_mutated(fixture).is_err());

        let (_, mut fixture) = committed_inputs();
        fixture["observations"][1]["observation_id"] = json!("fixture-observed-tc-onset");
        assert!(audit_mutated(fixture).is_err());
    }
}
