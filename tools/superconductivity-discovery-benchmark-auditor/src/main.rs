use jsonschema::validator_for;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn required_text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty field {key}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
}

fn held_out_contains(regime: &Value, field: &str, value: &str) -> bool {
    regime
        .get(field)
        .and_then(Value::as_array)
        .is_some_and(|values| values.iter().any(|candidate| candidate.as_str() == Some(value)))
}

fn expected_partition(regime: &Value, observation: &Value) -> Result<&'static str, String> {
    match required_text(regime, "kind")? {
        "family_held_out" => {
            let family = required_text(observation, "family_id")?;
            if regime["held_out_family_ids"].as_array().is_none_or(Vec::is_empty) {
                return Err("family split requires held_out_family_ids".to_owned());
            }
            Ok(if held_out_contains(regime, "held_out_family_ids", family) {
                "test"
            } else {
                "train"
            })
        }
        "source_paper_held_out" => {
            let source = required_text(observation, "source_id")?;
            if regime["held_out_source_ids"].as_array().is_none_or(Vec::is_empty) {
                return Err("source split requires held_out_source_ids".to_owned());
            }
            Ok(if held_out_contains(regime, "held_out_source_ids", source) {
                "test"
            } else {
                "train"
            })
        }
        "prototype_held_out" => {
            let prototype = required_text(observation, "prototype_id")?;
            if regime["held_out_prototype_ids"].as_array().is_none_or(Vec::is_empty) {
                return Err("prototype split requires held_out_prototype_ids".to_owned());
            }
            Ok(if held_out_contains(regime, "held_out_prototype_ids", prototype) {
                "test"
            } else {
                "train"
            })
        }
        "pressure_regime_held_out" => {
            let pressure = observation["pressure_gpa"]
                .as_f64()
                .ok_or_else(|| "pressure_gpa must be numeric".to_owned())?;
            let train_max = regime["train_pressure_max_gpa"]
                .as_f64()
                .ok_or_else(|| "pressure split requires train_pressure_max_gpa".to_owned())?;
            let test_min = regime["test_pressure_min_gpa"]
                .as_f64()
                .ok_or_else(|| "pressure split requires test_pressure_min_gpa".to_owned())?;
            if train_max >= test_min {
                return Err("pressure split requires train max below test min".to_owned());
            }
            Ok(if pressure <= train_max {
                "train"
            } else if pressure >= test_min {
                "test"
            } else {
                "excluded"
            })
        }
        "time_forward" => {
            let date = required_text(observation, "source_date")?;
            let train_end = required_text(regime, "train_source_date_end")?;
            let test_start = required_text(regime, "test_source_date_start")?;
            if !valid_iso_date(date) || !valid_iso_date(train_end) || !valid_iso_date(test_start) {
                return Err("time-forward dates must use YYYY-MM-DD".to_owned());
            }
            if train_end >= test_start {
                return Err("time split requires train cutoff before test start".to_owned());
            }
            Ok(if date <= train_end {
                "train"
            } else if date >= test_start {
                "test"
            } else {
                "excluded"
            })
        }
        other => Err(format!("unsupported split kind {other}")),
    }
}

fn group_value<'a>(observation: &'a Value, key: &str) -> Result<&'a str, String> {
    match key {
        "material_state_id" | "normalized_formula" | "family_id" | "source_id"
        | "prototype_id" => required_text(observation, key),
        other => Err(format!("unsupported group key {other}")),
    }
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
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
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "discovery benchmark schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture["contract"] != "SC-DISCOVERY-BENCHMARK-0.1" {
        return Err("unsupported discovery benchmark contract".to_owned());
    }

    let dataset = &fixture["dataset"];
    if required_text(dataset, "authority")? != "synthetic_test_fixture" {
        return Err(
            "initial auditor accepts only synthetic fixtures; external data need a separately reviewed import path"
                .to_owned(),
        );
    }
    let task = &fixture["task"];
    let target_property = required_text(task, "target_property")?;
    let target_unit = required_text(task, "target_unit")?;
    let task_authority = required_text(task, "label_authority")?;
    let allowed_measurement_classes: HashSet<&str> = task["measurement_classes"]
        .as_array()
        .ok_or_else(|| "measurement_classes must be an array".to_owned())?
        .iter()
        .map(|value| value.as_str().ok_or_else(|| "measurement class must be text".to_owned()))
        .collect::<Result<_, _>>()?;
    if allowed_measurement_classes.is_empty() {
        return Err("measurement_classes cannot be empty".to_owned());
    }

    for feature in task["available_features"]
        .as_array()
        .ok_or_else(|| "available_features must be an array".to_owned())?
    {
        let feature = feature
            .as_str()
            .ok_or_else(|| "feature names must be strings".to_owned())?;
        let normalized = feature.to_ascii_lowercase();
        if normalized == target_property
            || normalized == format!("measured_{target_property}")
            || normalized == format!("observed_{target_property}")
            || normalized == "target_value"
            || normalized == "source_id"
            || normalized == "source_paper_id"
            || normalized == "source_date"
            || normalized == "material_state_id"
            || normalized == "family_id"
            || normalized == "family_hypothesis"
        {
            return Err(format!(
                "target leakage: target property {target_property} appears in available features"
            ));
        }
    }

    let regimes = fixture["split_regimes"]
        .as_array()
        .ok_or_else(|| "split_regimes must be an array".to_owned())?;
    let observations = fixture["observations"]
        .as_array()
        .ok_or_else(|| "observations must be an array".to_owned())?;
    let record_count = dataset["record_count"]
        .as_u64()
        .ok_or_else(|| "dataset.record_count must be an integer".to_owned())?;
    if record_count as usize != observations.len() {
        return Err("dataset.record_count does not match observation count".to_owned());
    }

    let mut split_ids = HashSet::new();
    for regime in regimes {
        let id = required_text(regime, "split_id")?.to_owned();
        if !split_ids.insert(id.clone()) {
            return Err(format!("duplicate split ID {id}"));
        }
        let kind = required_text(regime, "kind")?;
        let keys = regime["group_keys"]
            .as_array()
            .ok_or_else(|| format!("split {id} group_keys must be an array"))?;
        let keys: HashSet<&str> = keys
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| format!("split {id} group key must be text"))
            })
            .collect::<Result<_, _>>()?;
        if !keys.contains("material_state_id") || !keys.contains("source_id") {
            return Err(format!(
                "split {id} must group material_state_id and source_id"
            ));
        }
        if kind == "family_held_out" && !keys.contains("family_id") {
            return Err(format!("family split {id} must group family_id"));
        }
        if kind == "prototype_held_out" && !keys.contains("prototype_id") {
            return Err(format!("prototype split {id} must group prototype_id"));
        }
    }

    let mut observation_ids = HashSet::new();
    let mut source_dates: HashMap<String, String> = HashMap::new();
    let mut group_partitions: HashMap<(String, String, String), String> = HashMap::new();
    let mut counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let mut measurement_class_counts: BTreeMap<String, u64> = BTreeMap::new();

    for observation in observations {
        let id = required_text(observation, "observation_id")?.to_owned();
        if !observation_ids.insert(id.clone()) {
            return Err(format!("duplicate observation_id {id}"));
        }
        if required_text(observation, "property_id")? != target_property {
            return Err(format!("observation {id} property does not match task target"));
        }
        if required_text(observation, "unit")? != target_unit {
            return Err(format!("observation {id} unit does not match task unit"));
        }
        let authority = required_text(observation, "label_authority")?;
        if authority != task_authority {
            return Err(format!(
                "mixed label authority: observation {id} is {authority}, task is {task_authority}"
            ));
        }
        let value = observation["value"]
            .as_f64()
            .ok_or_else(|| format!("observation {id} value must be numeric"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("observation {id} target must be finite and positive"));
        }

        let source_id = required_text(observation, "source_id")?.to_owned();
        let source_date = required_text(observation, "source_date")?.to_owned();
        if !valid_iso_date(&source_date) {
            return Err(format!("observation {id} source_date must use YYYY-MM-DD"));
        }
        if let Some(previous) = source_dates.get(&source_id) {
            if previous != &source_date {
                return Err(format!("source {source_id} has conflicting source dates"));
            }
        } else {
            source_dates.insert(source_id.clone(), source_date);
        }

        required_text(observation, "material_id")?;
        required_text(observation, "material_state_id")?;
        required_text(observation, "normalized_formula")?;
        required_text(observation, "family_id")?;
        required_text(observation, "source_locator")?;
        let measurement_class = required_text(observation, "measurement_class")?;
        if !allowed_measurement_classes.contains(measurement_class) {
            return Err(format!(
                "measurement class {measurement_class} is outside the preregistered task definition"
            ));
        }
        *measurement_class_counts
            .entry(measurement_class.to_owned())
            .or_insert(0) += 1;
        if task_authority == "measured_experiment"
            && required_text(observation, "measurement_method")? == "first_principles"
        {
            return Err(format!(
                "measured observation {id} cannot use first_principles as a measurement method"
            ));
        }
        if required_text(observation, "extraction_method")? != "synthetic_fixture" {
            return Err(format!(
                "synthetic-only fixture contains non-synthetic extraction method at {id}"
            ));
        }

        let assignments = observation["assignments"]
            .as_array()
            .ok_or_else(|| format!("observation {id} assignments must be an array"))?;
        if assignments.len() != regimes.len() {
            return Err(format!(
                "observation {id} has an incomplete split assignment set"
            ));
        }
        let mut seen_assignments = HashSet::new();
        for assignment in assignments {
            let split_id = required_text(assignment, "split_id")?;
            let partition = required_text(assignment, "partition")?;
            if !split_ids.contains(split_id) || !seen_assignments.insert(split_id.to_owned()) {
                return Err(format!("unknown or duplicate split assignment {split_id}"));
            }
            let regime = regimes
                .iter()
                .find(|regime| regime["split_id"].as_str() == Some(split_id))
                .ok_or_else(|| format!("missing split definition {split_id}"))?;
            let expected = expected_partition(regime, observation)?;
            if partition != expected {
                return Err(format!(
                    "split leakage or condition mismatch: observation {id}, split {split_id}, expected {expected}, received {partition}"
                ));
            }

            let counter = counts
                .entry(split_id.to_owned())
                .or_default()
                .entry(partition.to_owned())
                .or_insert(0);
            *counter += 1;

            for key in regime["group_keys"].as_array().unwrap() {
                let key = key.as_str().unwrap();
                let value = group_value(observation, key)?.to_owned();
                let map_key = (split_id.to_owned(), key.to_owned(), value);
                if let Some(previous) = group_partitions.get(&map_key) {
                    if previous != partition {
                        return Err(format!(
                            "group leakage: split {split_id}, key {key}, value {} appears in both {previous} and {partition}",
                            map_key.2
                        ));
                    }
                } else {
                    group_partitions.insert(map_key, partition.to_owned());
                }
            }
        }
        if seen_assignments.len() != split_ids.len() {
            return Err(format!("observation {id} is missing a split assignment"));
        }
    }

    let split_results = counts
        .into_iter()
        .map(|(split_id, partition_counts)| {
            let value = serde_json::to_value(partition_counts)
                .expect("integer split counts serialize");
            (split_id, value)
        })
        .collect::<serde_json::Map<String, Value>>();

    Ok(json!({
        "status": "pass",
        "contract": "SC-DISCOVERY-BENCHMARK-AUDIT-0.1",
        "benchmark_id": fixture["benchmark_id"],
        "dataset_authority": dataset["authority"],
        "dataset_version": dataset["dataset_version"],
        "synthetic_fixture_only": true,
        "external_data_loaded": false,
        "observation_count": observations.len(),
        "unique_observation_count": observation_ids.len(),
        "task": task,
        "split_results": split_results,
        "measurement_class_counts": measurement_class_counts,
        "model_metrics_status": "not_run_no_predictions_supplied",
        "schema_sha256": sha256_hex(schema_bytes),
        "fixture_sha256": sha256_hex(fixture_bytes),
        "claim_ceiling": fixture["claim_ceiling"]
    }))
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn usage(program: &str) {
    eprintln!("Usage: {program} --schema <schema.json> --fixture <fixture.json>");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" | "--fixture" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--schema" => schema_path = Some(path),
                    "--fixture" => fixture_path = Some(path),
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
    let receipt = audit(&read(&schema_path), &read(&fixture_path)).unwrap_or_else(|error| {
        eprintln!("SC-DISCOVERY-BENCHMARK-FAIL: {error}");
        process::exit(1);
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .expect("benchmark audit receipt serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (Vec<u8>, Vec<u8>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_discovery_benchmark_v0_1.schema.json"),
        )
        .unwrap();
        let fixture = fs::read(
            root.join("data/research/superconductivity_discovery_benchmark_fixture_v0_1.json"),
        )
        .unwrap();
        (schema, fixture)
    }

    fn rejects_mutation(mutate: impl FnOnce(&mut Value)) {
        let (schema, bytes) = inputs();
        let mut fixture: Value = serde_json::from_slice(&bytes).unwrap();
        mutate(&mut fixture);
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    fn set_partition(fixture: &mut Value, observation_id: &str, split_id: &str, partition: &str) {
        let observation = fixture["observations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|observation| observation["observation_id"] == observation_id)
            .unwrap();
        let assignment = observation["assignments"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|assignment| assignment["split_id"] == split_id)
            .unwrap();
        assignment["partition"] = json!(partition);
    }

    #[test]
    fn synthetic_fixture_passes_without_claiming_model_metrics() {
        let (schema, fixture) = inputs();
        let receipt = audit(&schema, &fixture).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["observation_count"], 7);
        assert_eq!(receipt["synthetic_fixture_only"], true);
        assert_eq!(
            receipt["model_metrics_status"],
            "not_run_no_predictions_supplied"
        );
        assert_eq!(
            receipt["split_results"]["family_holdout_cuprate"]["test"],
            2
        );
        assert_eq!(
            receipt["split_results"]["pressure_extrapolation_high"]["test"],
            1
        );
    }

    #[test]
    fn family_holdout_rejects_test_observation_in_training() {
        rejects_mutation(|fixture| {
            set_partition(
                fixture,
                "obs-ybco-tc-a",
                "family_holdout_cuprate",
                "train",
            );
        });
    }

    #[test]
    fn source_paper_holdout_rejects_test_source_in_training() {
        rejects_mutation(|fixture| {
            set_partition(
                fixture,
                "obs-ybco-tc-b",
                "source_holdout_ybco_paper",
                "train",
            );
        });
    }

    #[test]
    fn pressure_split_rejects_high_pressure_as_training() {
        rejects_mutation(|fixture| {
            set_partition(
                fixture,
                "obs-h3s-tc",
                "pressure_extrapolation_high",
                "train",
            );
        });
    }

    #[test]
    fn time_split_rejects_future_source_in_training() {
        rejects_mutation(|fixture| {
            set_partition(fixture, "obs-h3s-tc", "time_forward_2026", "train");
        });
    }

    #[test]
    fn mixed_measured_and_calculated_labels_are_rejected() {
        rejects_mutation(|fixture| {
            fixture["observations"][6]["label_authority"] =
                json!("calculated_first_principles");
        });
    }

    #[test]
    fn target_property_cannot_be_a_prediction_feature() {
        rejects_mutation(|fixture| {
            fixture["task"]["available_features"]
                .as_array_mut()
                .unwrap()
                .push(json!("measured_tc"));
        });
    }

    #[test]
    fn duplicate_observation_ids_are_rejected() {
        rejects_mutation(|fixture| {
            let duplicate = fixture["observations"][0].clone();
            fixture["observations"].as_array_mut().unwrap().push(duplicate);
            fixture["dataset"]["record_count"] = json!(8);
        });
    }

    #[test]
    fn measurement_class_must_be_preregistered_for_the_task() {
        rejects_mutation(|fixture| {
            fixture["observations"][0]["measurement_class"] = json!("spectroscopic_gap");
        });
    }

    #[test]
    fn source_identifier_cannot_be_a_prediction_feature() {
        rejects_mutation(|fixture| {
            fixture["task"]["available_features"]
                .as_array_mut()
                .unwrap()
                .push(json!("source_id"));
        });
    }

    #[test]
    fn family_identity_cannot_be_a_prediction_feature() {
        rejects_mutation(|fixture| {
            fixture["task"]["available_features"]
                .as_array_mut()
                .unwrap()
                .push(json!("family_id"));
        });
    }

    #[test]
    fn malformed_source_dates_are_rejected() {
        rejects_mutation(|fixture| {
            fixture["observations"][6]["source_date"] = json!("2026/02/03");
        });
    }

    #[test]
    fn group_leakage_is_rejected_even_when_declared_partitions_match() {
        rejects_mutation(|fixture| {
            fixture["observations"][4]["normalized_formula"] = json!("Pb");
        });
    }
}
