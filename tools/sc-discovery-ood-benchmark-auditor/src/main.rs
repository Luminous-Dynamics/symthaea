use jsonschema::validator_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
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

fn strings(value: &Value, key: &str) -> Result<Vec<String>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} contains a non-string value"))
        })
        .collect()
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn group_value<'a>(record: &'a Value, key: &str) -> Result<&'a str, String> {
    match key {
        "family_id" | "prototype_id" | "source_id" => required_text(record, key),
        _ => Err(format!("unsupported group key {key}")),
    }
}

fn check_schema(schema_bytes: &[u8], fixture: &Value) -> Result<(), String> {
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(fixture).next() {
        return Err(format!(
            "manifest schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    Ok(())
}

fn record_map<'a>(records: &'a [Value]) -> Result<HashMap<&'a str, &'a Value>, String> {
    let mut map = HashMap::new();
    let mut identities = HashSet::new();
    for record in records {
        let id = required_text(record, "record_id")?;
        if map.insert(id, record).is_some() {
            return Err(format!("duplicate record_id {id}"));
        }
        let identity = [
            required_text(record, "material_id")?,
            required_text(record, "property")?,
            required_text(record, "source_id")?,
            required_text(record, "publication_date")?,
            required_text(record, "specimen_id")?,
            required_text(record, "tc_criterion")?,
        ]
        .join("|");
        if !identities.insert(identity) {
            return Err(format!(
                "duplicate material/property/source/specimen identity for {id}"
            ));
        }
        let value = record["value"]
            .as_f64()
            .ok_or_else(|| format!("record {id} has no numeric value"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!("record {id} has invalid non-negative measurement value"));
        }
    }
    Ok(map)
}

fn id_set(split: &Value, key: &str) -> Result<HashSet<String>, String> {
    let values = strings(split, key)?;
    let set: HashSet<String> = values.iter().cloned().collect();
    if set.len() != values.len() {
        return Err(format!("{key} contains duplicate IDs"));
    }
    Ok(set)
}

fn audit_split(
    split: &Value,
    records: &HashMap<&str, &Value>,
    evidence_class: &str,
) -> Result<(HashSet<String>, HashSet<String>, HashSet<String>), String> {
    let split_id = required_text(split, "split_id")?;
    let train = id_set(split, "train_record_ids")?;
    let calibration = id_set(split, "calibration_record_ids")?;
    let test = id_set(split, "test_record_ids")?;
    if train.is_empty() || test.is_empty() {
        return Err(format!("split {split_id} needs non-empty train and test sets"));
    }
    if !train.is_disjoint(&calibration)
        || !train.is_disjoint(&test)
        || !calibration.is_disjoint(&test)
    {
        return Err(format!("split {split_id} assigns a record to multiple roles"));
    }
    let all: HashSet<String> = train
        .union(&calibration)
        .cloned()
        .chain(test.iter().cloned())
        .collect();
    for id in &all {
        let record = records
            .get(id.as_str())
            .ok_or_else(|| format!("split {split_id} references unknown record {id}"))?;
        if required_text(record, "evidence_class")? != evidence_class {
            return Err(format!("split {split_id} mixes evidence class at record {id}"));
        }
        if required_text(record, "property")? != required_text(split, "property")? {
            return Err(format!("split {split_id} mixes target properties at record {id}"));
        }
        if required_text(record, "unit")? != required_text(split, "unit")? {
            return Err(format!("split {split_id} mixes target units at record {id}"));
        }
    }

    match required_text(split, "mode")? {
        "group_holdout" => {
            let key = required_text(split, "group_key")?;
            let held_out: HashSet<String> = strings(split, "held_out_values")?
                .into_iter()
                .collect();
            if held_out.is_empty() {
                return Err(format!("split {split_id} has no held-out group values"));
            }
            let mut test_values = HashSet::new();
            let mut train_values = HashSet::new();
            for id in &test {
                let value = group_value(records[id.as_str()], key)?.to_owned();
                if !held_out.contains(&value) {
                    return Err(format!(
                        "split {split_id} test record {id} is outside declared held-out groups"
                    ));
                }
                test_values.insert(value);
            }
            for id in train.union(&calibration) {
                train_values.insert(group_value(records[id.as_str()], key)?.to_owned());
            }
            if !train_values.is_disjoint(&test_values) {
                return Err(format!(
                    "split {split_id} leaks {key} between train/calibration and test"
                ));
            }
            if !held_out.is_subset(&test_values) {
                return Err(format!(
                    "split {split_id} declares held-out groups with no test records"
                ));
            }
        }
        "temporal_holdout" => {
            let cutoff = required_text(split, "cutoff_date")?;
            for id in train.union(&calibration) {
                let date = required_text(records[id.as_str()], "publication_date")?;
                if date >= cutoff {
                    return Err(format!(
                        "split {split_id} has non-training-era record {id} at/after cutoff"
                    ));
                }
            }
            for id in &test {
                let date = required_text(records[id.as_str()], "publication_date")?;
                if date < cutoff {
                    return Err(format!("split {split_id} has pre-cutoff test record {id}"));
                }
            }
        }
        other => return Err(format!("split {split_id} has unsupported mode {other}")),
    }
    Ok((train, calibration, test))
}

fn summarize(
    split_id: &str,
    model_id: &str,
    test: &HashSet<String>,
    records: &HashMap<&str, &Value>,
    predictions: &[Value],
    property: &str,
) -> Result<Value, String> {
    let relevant: Vec<&Value> = predictions
        .iter()
        .filter(|prediction| {
            prediction["split_id"].as_str() == Some(split_id)
                && prediction["model_id"].as_str() == Some(model_id)
                && prediction["property"].as_str() == Some(property)
        })
        .collect();
    if relevant.len() != test.len() {
        return Err(format!(
            "split {split_id}, model {model_id}: expected {} test predictions, found {}",
            test.len(),
            relevant.len()
        ));
    }
    let mut seen = HashSet::new();
    let mut absolute_errors = Vec::new();
    let mut relative_errors = Vec::new();
    let mut log_errors = Vec::new();
    let mut log_pairs = Vec::new();
    let mut interval_hits = 0usize;
    let mut interval_count = 0usize;
    let mut covered_count = 0usize;
    for prediction in relevant {
        let record_id = required_text(prediction, "record_id")?;
        if !test.contains(record_id) || !seen.insert(record_id.to_owned()) {
            return Err(format!(
                "prediction for {record_id} is duplicate or not a test record"
            ));
        }
        let abstained = prediction["abstained"]
            .as_bool()
            .ok_or_else(|| "abstained must be boolean".to_owned())?;
        let predicted = prediction["predicted_value"].as_f64();
        let lower = prediction["lower"].as_f64();
        let upper = prediction["upper"].as_f64();
        if abstained {
            if predicted.is_some() || lower.is_some() || upper.is_some() {
                return Err(format!(
                    "abstaining prediction {record_id} must not carry a point or interval estimate"
                ));
            }
            continue;
        }
        let predicted = predicted
            .ok_or_else(|| format!("non-abstaining prediction {record_id} lacks a point estimate"))?;
        if !predicted.is_finite() || predicted < 0.0 {
            return Err(format!("prediction {record_id} has invalid value"));
        }
        if lower.is_some() != upper.is_some() {
            return Err(format!(
                "prediction {record_id} must provide both interval bounds or neither"
            ));
        }
        if let (Some(lo), Some(hi)) = (lower, upper) {
            if !lo.is_finite() || !hi.is_finite() || lo > hi || lo < 0.0 {
                return Err(format!("prediction {record_id} has invalid interval"));
            }
        }
        let observed = records[record_id]
            .get("value")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("test record {record_id} has no target value"))?;
        let absolute_error = (predicted - observed).abs();
        absolute_errors.push(absolute_error);
        if observed > 0.0 {
            relative_errors.push(absolute_error / observed);
        }
        if predicted > 0.0 && observed > 0.0 {
            let predicted_log = predicted.log10();
            let observed_log = observed.log10();
            log_errors.push((predicted_log - observed_log).abs());
            log_pairs.push((predicted_log, observed_log));
        }
        if let (Some(lo), Some(hi)) = (lower, upper) {
            interval_count += 1;
            if observed >= lo && observed <= hi {
                interval_hits += 1;
            }
        }
        covered_count += 1;
    }
    if seen.len() != test.len() {
        return Err(format!(
            "split {split_id}, model {model_id} is missing test predictions"
        ));
    }
    let mae = if absolute_errors.is_empty() {
        None
    } else {
        Some(absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64)
    };
    let log_mae = if log_errors.is_empty() {
        None
    } else {
        Some(log_errors.iter().sum::<f64>() / log_errors.len() as f64)
    };
    relative_errors.sort_by(f64::total_cmp);
    let median_relative_error = match relative_errors.len() {
        0 => None,
        count if count % 2 == 1 => Some(relative_errors[count / 2]),
        count => Some((relative_errors[count / 2 - 1] + relative_errors[count / 2]) / 2.0),
    };
    let r2_log = if log_pairs.len() < 2 {
        None
    } else {
        let mean_observed = log_pairs.iter().map(|(_, observed)| observed).sum::<f64>()
            / log_pairs.len() as f64;
        let residual_sum = log_pairs.iter()
            .map(|(predicted, observed)| (predicted - observed).powi(2))
            .sum::<f64>();
        let total_sum = log_pairs.iter()
            .map(|(_, observed)| (observed - mean_observed).powi(2))
            .sum::<f64>();
        if total_sum <= f64::EPSILON {
            None
        } else {
            Some(1.0 - residual_sum / total_sum)
        }
    };
    Ok(json!({
        "model_id": model_id,
        "test_count": test.len(),
        "predicted_count": covered_count,
        "coverage": covered_count as f64 / test.len() as f64,
        "abstention_count": test.len() - covered_count,
        "mae": mae,
        "median_absolute_relative_error": median_relative_error,
        "mean_absolute_log10_error": log_mae,
        "r2_log": r2_log,
        "interval_count": interval_count,
        "empirical_interval_coverage": if interval_count == 0 {
            None::<f64>
        } else {
            Some(interval_hits as f64 / interval_count as f64)
        },
        "interpretation": "fixture_metric_only; never report as measured or SOTA performance"
    }))
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    check_schema(schema_bytes, &fixture)?;
    if fixture["contract"] != "SC-DISCOVERY-OOD-BENCHMARK-0.1" {
        return Err("unsupported SC discovery benchmark contract".to_owned());
    }
    let data_status = required_text(&fixture["dataset"], "data_status")?;
    let evidence_class = required_text(&fixture["protocol"], "evaluation_evidence_class")?;
    if data_status == "synthetic_fixture" && evidence_class != "synthetic_fixture" {
        return Err("synthetic dataset must remain in the synthetic evidence tier".to_owned());
    }
    if data_status == "curated_external" && evidence_class == "synthetic_fixture" {
        return Err("curated external dataset cannot be labelled synthetic evidence".to_owned());
    }
    if fixture["protocol"]["reject_random_split"].as_bool() != Some(true) {
        return Err("discovery protocol must reject random row splits".to_owned());
    }

    let records = fixture["records"]
        .as_array()
        .ok_or_else(|| "records must be an array".to_owned())?;
    let record_index = record_map(records)?;
    let splits = fixture["splits"]
        .as_array()
        .ok_or_else(|| "splits must be an array".to_owned())?;
    let predictions = fixture["predictions"]
        .as_array()
        .ok_or_else(|| "predictions must be an array".to_owned())?;
    let models = fixture["models"]
        .as_array()
        .ok_or_else(|| "models must be an array".to_owned())?;
    let model_ids: HashSet<String> = models
        .iter()
        .map(|model| required_text(model, "model_id").map(str::to_owned))
        .collect::<Result<_, _>>()?;
    if model_ids.len() != models.len() {
        return Err("model_id values must be unique".to_owned());
    }
    let required_keys = strings(&fixture["protocol"], "required_group_keys")?;
    for key in &required_keys {
        if !matches!(key.as_str(), "family_id" | "prototype_id" | "source_id") {
            return Err(format!("unsupported required group key {key}"));
        }
    }
    let mut split_ids = HashSet::new();
    let mut split_test_ids: HashMap<String, HashSet<String>> = HashMap::new();
    let mut reports = Vec::new();
    for split in splits {
        let split_id = required_text(split, "split_id")?;
        if !split_ids.insert(split_id.to_owned()) {
            return Err(format!("duplicate split_id {split_id}"));
        }
        let mode = required_text(split, "mode")?;
        if mode == "group_holdout" {
            let key = required_text(split, "group_key")?;
            if !required_keys.iter().any(|required| required == key) {
                return Err(format!(
                    "split {split_id} uses group key {key} not required by protocol"
                ));
            }
        }
        let (train, calibration, test) = audit_split(split, &record_index, evidence_class)?;
        split_test_ids.insert(split_id.to_owned(), test.clone());
        let property = required_text(split, "property")?;
        let mut sorted_model_ids: Vec<String> = model_ids.iter().cloned().collect();
        sorted_model_ids.sort();
        let mut model_reports = Vec::new();
        for model_id in &sorted_model_ids {
            model_reports.push(summarize(
                split_id,
                model_id,
                &test,
                &record_index,
                predictions,
                property,
            )?);
        }
        reports.push(json!({
            "split_id": split_id,
            "mode": mode,
            "group_key": split.get("group_key").cloned().unwrap_or(Value::Null),
            "train_count": train.len(),
            "calibration_count": calibration.len(),
            "test_count": test.len(),
            "property": property,
            "unit": split["unit"],
            "model_metrics": model_reports
        }));
    }
    for prediction in predictions {
        let split_id = required_text(prediction, "split_id")?;
        if !split_ids.contains(split_id) {
            return Err(format!("prediction references unknown split {split_id}"));
        }
        let model_id = required_text(prediction, "model_id")?;
        if !model_ids.contains(model_id) {
            return Err(format!("prediction references unknown model {model_id}"));
        }
        let record_id = required_text(prediction, "record_id")?;
        let record = record_index.get(record_id)
            .ok_or_else(|| format!("prediction references unknown record {record_id}"))?;
        let split = splits.iter().find(|split| split["split_id"].as_str() == Some(split_id))
            .ok_or_else(|| format!("prediction references unknown split {split_id}"))?;
        let expected_property = required_text(split, "property")?;
        if required_text(prediction, "property")? != expected_property
            || required_text(record, "property")? != expected_property
        {
            return Err(format!("prediction {record_id} does not match split {split_id} property"));
        }
        if !split_test_ids.get(split_id).is_some_and(|test| test.contains(record_id)) {
            return Err(format!("prediction {record_id} is not part of split {split_id} test set"));
        }
    }
    Ok(json!({
        "contract": "SC-DISCOVERY-OOD-BENCHMARK-0.1",
        "dataset_id": fixture["dataset"]["dataset_id"],
        "dataset_revision": fixture["dataset"]["revision"],
        "data_status": data_status,
        "evidence_class": evidence_class,
        "records_audited": records.len(),
        "splits_audited": reports.len(),
        "model_count": model_ids.len(),
        "split_reports": reports,
        "claim_ceiling": if data_status == "synthetic_fixture" {
            "synthetic_fixture_software_validation_only"
        } else {
            "curated_dataset_audit_only_no_experimental_validation_claim"
        },
        "schema_sha256": sha256(schema_bytes),
        "fixture_sha256": sha256(fixture_bytes)
    }))
}

fn inputs() -> (Vec<u8>, Vec<u8>) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let schema = fs::read(root.join(
        "data/schemas/sc_discovery_benchmark_v0_1.schema.json"
    )).unwrap();
    let fixture = fs::read(root.join(
        "data/research/sc_discovery_benchmark_fixture_v0_1.json"
    )).unwrap();
    (schema, fixture)
}

fn main() {
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
    let args: Vec<String> = env::args().collect();
    let mut index = 1;
    while index < args.len() {
        let flag = args[index].as_str();
        if index + 1 >= args.len() {
            eprintln!("missing value for {flag}");
            process::exit(2);
        }
        let value = args[index + 1].clone();
        match flag {
            "--schema" => schema_path = Some(PathBuf::from(value)),
            "--fixture" => fixture_path = Some(PathBuf::from(value)),
            "--out" => output_path = Some(PathBuf::from(value)),
            _ => {
                eprintln!("unknown argument {flag}");
                process::exit(2);
            }
        }
        index += 2;
    }
    let schema_path = schema_path.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/schemas/sc_discovery_benchmark_v0_1.schema.json")
    });
    let fixture_path = fixture_path.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/research/sc_discovery_benchmark_fixture_v0_1.json")
    });
    let schema_bytes = fs::read(&schema_path).unwrap_or_else(|error| {
        eprintln!("cannot read schema {}: {error}", schema_path.display());
        process::exit(2);
    });
    let fixture_bytes = fs::read(&fixture_path).unwrap_or_else(|error| {
        eprintln!("cannot read fixture {}: {error}", fixture_path.display());
        process::exit(2);
    });
    let receipt = audit(&schema_bytes, &fixture_bytes).unwrap_or_else(|error| {
        eprintln!("SC-DISCOVERY-OOD-FAIL: {error}");
        process::exit(1);
    });
    let bytes = serde_json::to_vec_pretty(&receipt).expect("receipt serializes");
    if let Some(path) = output_path {
        fs::write(&path, &bytes).unwrap_or_else(|error| {
            eprintln!("cannot write receipt {}: {error}", path.display());
            process::exit(2);
        });
    }
    println!("{}", String::from_utf8_lossy(&bytes));
}

#[cfg(test)]
mod tests {
    use super::audit;
    use serde_json::Value;

    fn inputs() -> (Vec<u8>, Value) {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = std::fs::read(root.join(
            "data/schemas/sc_discovery_benchmark_v0_1.schema.json"
        )).unwrap();
        let fixture_bytes = std::fs::read(root.join(
            "data/research/sc_discovery_benchmark_fixture_v0_1.json"
        )).unwrap();
        (schema, serde_json::from_slice(&fixture_bytes).unwrap())
    }

    #[test]
    fn synthetic_fixture_passes_and_never_claims_physical_evidence() {
        let (schema, fixture) = inputs();
        let bytes = serde_json::to_vec(&fixture).unwrap();
        let receipt = audit(&schema, &bytes).unwrap();
        assert_eq!(receipt["data_status"], "synthetic_fixture");
        assert_eq!(receipt["claim_ceiling"], "synthetic_fixture_software_validation_only");
        assert_eq!(receipt["splits_audited"], 4);
    }

    #[test]
    fn family_overlap_is_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["records"][0]["family_id"] = Value::String("iron_pnictide".to_owned());
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn source_overlap_is_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["records"][0]["source_id"] =
            Value::String("src-pnictide-la1111".to_owned());
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn temporal_leakage_is_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["records"][0]["publication_date"] =
            Value::String("2030-01-01".to_owned());
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn evidence_tier_mixing_is_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["records"][4]["evidence_class"] = Value::String("computed".to_owned());
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn missing_test_prediction_is_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["predictions"].as_array_mut().unwrap().pop();
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn abstention_must_not_smuggle_a_prediction() {
        let (schema, mut fixture) = inputs();
        fixture["predictions"][1]["abstained"] = Value::Bool(true);
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn duplicate_record_ids_are_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["records"][1]["record_id"] = fixture["records"][0]["record_id"].clone();
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn prediction_for_training_record_is_rejected() {
        let (schema, mut fixture) = inputs();
        fixture["predictions"][0]["record_id"] = Value::String("syn-al-tc".to_owned());
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn prediction_property_must_match_split_target() {
        let (schema, mut fixture) = inputs();
        fixture["predictions"][0]["property"] = Value::String("Hc2".to_owned());
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }

    #[test]
    fn random_row_split_cannot_be_enabled() {
        let (schema, mut fixture) = inputs();
        fixture["protocol"]["reject_random_split"] = Value::Bool(false);
        assert!(audit(&schema, &serde_json::to_vec(&fixture).unwrap()).is_err());
    }
}
