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
        .ok_or_else(|| format!("missing or empty text field {key}"))
}

fn validate_dataset(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
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
            "observation-inverse schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if fixture.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("initial inverse-observation auditor accepts synthetic fixtures only".to_owned());
    }
    if required_text(&fixture, "contract")? != "SC-OBSERVATION-INVERSE-0.1" {
        return Err("unsupported observation-inverse contract".to_owned());
    }

    let dataset_id = required_text(&fixture, "dataset_id")?;
    let material = fixture
        .get("material")
        .ok_or_else(|| "missing material identity".to_owned())?;
    let specimen = fixture
        .get("specimen")
        .ok_or_else(|| "missing specimen identity".to_owned())?;
    let material_id = required_text(material, "material_id")?;
    let specimen_id = required_text(specimen, "specimen_id")?;
    let batch_id = required_text(specimen, "batch_id")?;
    let sample_state_id = required_text(specimen, "sample_state_id")?;
    let region_id = required_text(specimen, "region_id")?;
    let _ = required_text(material, "nominal_composition")?;
    let _ = required_text(material, "structure_identity")?;
    let _ = required_text(specimen, "sample_form")?;

    let _process_history = fixture
        .get("process_history")
        .ok_or_else(|| "missing process_history".to_owned())?;

    let observations = fixture
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "observations must be an array".to_owned())?;
    if observations.len() < 2 {
        return Err("at least two modality observations are required".to_owned());
    }

    let mut observation_ids = HashSet::with_capacity(observations.len());
    let mut raw_hashes = HashSet::with_capacity(observations.len());
    let mut observation_modalities = HashMap::with_capacity(observations.len());
    let mut condition_variants = HashSet::new();
    let mut observation_count_by_modality: HashMap<String, usize> = HashMap::new();

    for (index, observation) in observations.iter().enumerate() {
        let id = required_text(observation, "observation_id")?;
        if !observation_ids.insert(id.to_owned()) {
            return Err(format!("duplicate observation_id {id}"));
        }
        let modality = required_text(observation, "modality")?;
        observation_modalities.insert(id.to_owned(), modality.to_owned());
        *observation_count_by_modality
            .entry(modality.to_owned())
            .or_insert(0) += 1;

        let measurand = required_text(observation, "measurand")?;
        let unit = required_text(observation, "unit")?;
        let value = observation
            .get("value")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("observation {id} value must be numeric"))?;
        let uncertainty = observation
            .get("standard_uncertainty")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("observation {id} standard_uncertainty must be numeric"))?;
        if !value.is_finite() || !uncertainty.is_finite() || uncertainty < 0.0 {
            return Err(format!("observation {id} has non-finite value or invalid uncertainty"));
        }
        let _ = required_text(observation, "measurement_scope")?;
        let _ = required_text(observation, "method_id")?;
        let _ = required_text(observation, "calibration_id")?;
        let _ = required_text(observation, "source_reference")?;

        let raw_hash = required_text(observation, "raw_data_sha256")?;
        if raw_hash.len() != 64
            || !raw_hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("observation {id} raw_data_sha256 must be lowercase hex SHA-256"));
        }
        raw_hashes.insert(raw_hash.to_owned());

        let conditions = observation
            .get("conditions")
            .ok_or_else(|| format!("observation {id} missing conditions"))?;
        let condition_temperature = conditions
            .get("temperature")
            .ok_or_else(|| format!("observation {id} missing temperature"))?;
        if !condition_temperature.is_null() {
            let temperature = condition_temperature
                .as_f64()
                .ok_or_else(|| format!("observation {id} temperature must be numeric or null"))?;
            if !temperature.is_finite() || temperature < 0.0 {
                return Err(format!("observation {id} invalid temperature"));
            }
        }
        let temperature_unit = required_text(conditions, "temperature_unit")?;
        let field_unit = required_text(conditions, "field_unit")?;
        let field_orientation = required_text(conditions, "field_orientation")?;
        let pressure_unit = required_text(conditions, "pressure_unit")?;
        let geometry_id = required_text(conditions, "geometry_id")?;
        let timestamp = required_text(conditions, "measurement_timestamp")?;
        let _ = (temperature_unit, field_unit, field_orientation, pressure_unit, geometry_id, timestamp);

        // Keep condition variants explicit rather than requiring identical conditions
        // across distinct modalities on the same specimen.
        let condition_fingerprint = serde_json::to_string(conditions)
            .map_err(|error| format!("could not canonicalize conditions for {id}: {error}"))?;
        condition_variants.insert(condition_fingerprint);

        // The top-level region_id binds observations to a specimen region.
        // geometry_id remains observation-specific because instruments and
        // modalities may have different measurement geometries.

        let _ = (measurand, unit, index);
    }

    let split = fixture
        .get("split")
        .ok_or_else(|| "missing calibration/holdout split".to_owned())?;
    let split_id = required_text(split, "split_id")?;
    if required_text(split, "holdout_rule")? != "leave_one_modality_out" {
        return Err("initial auditor supports leave_one_modality_out splits only".to_owned());
    }
    let calibration_ids = split
        .get("calibration_observation_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "calibration_observation_ids must be an array".to_owned())?;
    let heldout_ids = split
        .get("heldout_observation_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "heldout_observation_ids must be an array".to_owned())?;
    if calibration_ids.is_empty() || heldout_ids.is_empty() {
        return Err("calibration and held-out partitions must both be non-empty".to_owned());
    }

    let mut calibration_set = HashSet::with_capacity(calibration_ids.len());
    let mut heldout_set = HashSet::with_capacity(heldout_ids.len());
    let mut calibration_modalities = HashSet::new();
    let mut heldout_modalities = HashSet::new();

    for value in calibration_ids {
        let id = value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "calibration observation IDs must be non-empty strings".to_owned())?;
        if !calibration_set.insert(id.to_owned()) {
            return Err(format!("duplicate calibration observation ID {id}"));
        }
        let modality = observation_modalities
            .get(id)
            .ok_or_else(|| format!("calibration split references unknown observation {id}"))?;
        calibration_modalities.insert(modality.clone());
    }
    for value in heldout_ids {
        let id = value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "held-out observation IDs must be non-empty strings".to_owned())?;
        if !heldout_set.insert(id.to_owned()) {
            return Err(format!("duplicate held-out observation ID {id}"));
        }
        if calibration_set.contains(id) {
            return Err(format!("observation {id} appears in both calibration and held-out partitions"));
        }
        let modality = observation_modalities
            .get(id)
            .ok_or_else(|| format!("held-out split references unknown observation {id}"))?;
        heldout_modalities.insert(modality.clone());
    }

    if calibration_set.len() + heldout_set.len() != observation_ids.len()
        || observation_ids
            .iter()
            .any(|id| !calibration_set.contains(id) && !heldout_set.contains(id))
    {
        return Err("calibration and held-out sets must partition every observation exactly once".to_owned());
    }
    if heldout_modalities.len() != 1 {
        return Err("leave_one_modality_out requires exactly one held-out modality".to_owned());
    }
    if heldout_modalities
        .iter()
        .any(|modality| calibration_modalities.contains(modality))
    {
        return Err("held-out modality is present in calibration data (modality leakage)".to_owned());
    }

    let model_target = fixture
        .get("model_target")
        .ok_or_else(|| "missing model_target".to_owned())?;
    let model_family = required_text(model_target, "model_family")?;
    let model_version = required_text(model_target, "model_version")?;
    let prior_id = required_text(model_target, "prior_definition_id")?;
    let operator_version = required_text(model_target, "observation_operator_version")?;
    let target_parameters = model_target
        .get("target_parameters")
        .and_then(Value::as_array)
        .ok_or_else(|| "target_parameters must be an array".to_owned())?;
    if target_parameters.is_empty() {
        return Err("at least one target parameter is required".to_owned());
    }
    let mut parameter_ids = HashSet::new();
    for value in target_parameters {
        let parameter = value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "target parameter IDs must be non-empty strings".to_owned())?;
        if !parameter_ids.insert(parameter.to_owned()) {
            return Err(format!("duplicate target parameter {parameter}"));
        }
    }

    let schema_sha256 = format!("{:x}", Sha256::digest(schema_bytes));
    let fixture_sha256 = format!("{:x}", Sha256::digest(fixture_bytes));
    Ok(json!({
        "status": "pass",
        "contract": "SC-OBSERVATION-INVERSE-AUDIT-0.1",
        "dataset_id": dataset_id,
        "schema_sha256": schema_sha256,
        "fixture_sha256": fixture_sha256,
        "specimen": {
            "material_id": material_id,
            "specimen_id": specimen_id,
            "batch_id": batch_id,
            "sample_state_id": sample_state_id,
            "region_id": region_id
        },
        "observation_count": observations.len(),
        "unique_raw_data_digest_count": raw_hashes.len(),
        "modality_count": observation_count_by_modality.len(),
        "modalities": observation_count_by_modality,
        "condition_variant_count": condition_variants.len(),
        "split": {
            "split_id": split_id,
            "calibration_observation_count": calibration_set.len(),
            "heldout_observation_count": heldout_set.len(),
            "calibration_modalities": calibration_modalities,
            "heldout_modalities": heldout_modalities,
            "partition_complete": true,
            "modality_leakage": false
        },
        "model_target": {
            "model_family": model_family,
            "model_version": model_version,
            "target_parameter_count": parameter_ids.len(),
            "prior_definition_id": prior_id,
            "observation_operator_version": operator_version
        },
        "claim_ceiling": "Specimen-linked observation/split audit only. No inverse solution, posterior, calibrated material property or Tc is established."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --schema <schema.json> --fixture <specimen-observations.json> [--out <receipt.json>]"
    );
}

fn read(path: &PathBuf) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        eprintln!("could not read {}: {error}", path.display());
        process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut schema_path = None;
    let mut fixture_path = None;
    let mut output_path = None;
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
    let receipt = validate_dataset(&read(&schema_path), &read(&fixture_path))
        .unwrap_or_else(|error| {
            eprintln!("SC-OBSERVATION-INVERSE-FAIL: {error}");
            process::exit(1);
        });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing inverse-observation receipt should not fail");
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
    use std::path::PathBuf;

    fn inputs() -> (Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let schema = fs::read(
            root.join("data/schemas/superconductivity_observation_inverse_instance_v0_1.schema.json"),
        )
        .expect("inverse observation schema must be readable");
        let fixture: Value = serde_json::from_slice(
            &fs::read(root.join("data/research/superconductivity_observation_inverse_fixture_v0_1.json"))
                .expect("inverse observation fixture must be readable"),
        )
        .expect("inverse observation fixture must parse");
        (schema, fixture)
    }

    fn audit_mutated(fixture: Value) -> Result<Value, String> {
        let (schema, _) = inputs();
        validate_dataset(&schema, &serde_json::to_vec(&fixture).unwrap())
    }

    #[test]
    fn synthetic_same_specimen_leave_modality_out_split_passes() {
        let (_, fixture) = inputs();
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["observation_count"], 3);
        assert_eq!(receipt["split"]["calibration_observation_count"], 2);
        assert_eq!(receipt["split"]["heldout_observation_count"], 1);
        assert_eq!(receipt["split"]["modality_leakage"], false);
    }

    #[test]
    fn calibration_and_heldout_overlap_fails_closed() {
        let (_, mut fixture) = inputs();
        fixture["split"]["heldout_observation_ids"] = json!(["obs-amplitude-001"]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn holdout_modality_cannot_appear_in_calibration() {
        let (_, mut fixture) = inputs();
        fixture["observations"][2]["modality"] =
            json!("heat_capacity");
        fixture["split"]["calibration_observation_ids"] =
            json!(["obs-amplitude-001", "obs-vorticity-001"]);
        fixture["split"]["heldout_observation_ids"] =
            json!(["obs-heat-capacity-001"]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn every_observation_must_be_assigned_exactly_once() {
        let (_, mut fixture) = inputs();
        fixture["split"]["calibration_observation_ids"] =
            json!(["obs-amplitude-001"]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn unknown_observation_id_is_rejected() {
        let (_, mut fixture) = inputs();
        fixture["split"]["heldout_observation_ids"] = json!(["missing-observation"]);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn duplicate_observation_identity_is_rejected() {
        let (_, mut fixture) = inputs();
        fixture["observations"][1]["observation_id"] = json!("obs-amplitude-001");
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn shared_raw_data_digest_is_allowed_for_derived_observations() {
        let (_, mut fixture) = inputs();
        let shared_digest = fixture["observations"][0]["raw_data_sha256"].clone();
        fixture["observations"][1]["raw_data_sha256"] = shared_digest;
        let receipt = audit_mutated(fixture).unwrap();
        assert_eq!(receipt["observation_count"], 3);
        assert_eq!(receipt["unique_raw_data_digest_count"], 2);
    }

    #[test]
    fn observation_specific_geometry_ids_can_differ_within_one_region() {
        let (_, mut fixture) = inputs();
        fixture["observations"][1]["conditions"]["geometry_id"] = json!("instrument-geometry-b");
        assert!(audit_mutated(fixture).is_ok());
    }

    #[test]
    fn invalid_uncertainty_is_rejected() {
        let (_, mut fixture) = inputs();
        fixture["observations"][0]["standard_uncertainty"] = json!(-0.1);
        assert!(audit_mutated(fixture).is_err());
    }

    #[test]
    fn leave_one_modality_out_requires_one_unseen_modality() {
        let (_, mut fixture) = inputs();
        fixture["observations"][2]["modality"] =
            json!("amplitude_sensitive_spectroscopy");
        assert!(audit_mutated(fixture).is_err());
    }
}
