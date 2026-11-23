use jsonschema::validator_for;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn nonempty_text<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("missing or empty string field {field}"))
}

fn validate_document(
    schema_bytes: &[u8],
    document_bytes: &[u8],
    label: &str,
) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("{label} schema JSON parse failed: {error}"))?;
    let document: Value = serde_json::from_slice(document_bytes)
        .map_err(|error| format!("{label} JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("{label} schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|error| format!("{label} schema compile failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&document).next() {
        return Err(format!(
            "{label} schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    Ok(document)
}

fn validate_comparison(
    comparison_schema_bytes: &[u8],
    comparison_bytes: &[u8],
    observation_schema_bytes: &[u8],
    observation_fixture_bytes: &[u8],
) -> Result<Value, String> {
    let comparison = validate_document(comparison_schema_bytes, comparison_bytes, "comparison")?;
    let fixture: Value = serde_json::from_slice(observation_fixture_bytes)
        .map_err(|error| format!("observation fixture JSON parse failed: {error}"))?;
    let observation_schema: Value = serde_json::from_slice(observation_schema_bytes)
        .map_err(|error| format!("observation schema JSON parse failed: {error}"))?;
    const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
    if observation_schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("observation schema must declare {DIALECT}"));
    }
    let observation_validator = validator_for(&observation_schema)
        .map_err(|error| format!("observation schema compile failed: {error}"))?;

    if comparison.get("fixture_only").and_then(Value::as_bool) != Some(true) {
        return Err("this executable accepts synthetic comparison fixtures only".to_owned());
    }

    let source_ids = fixture
        .get("fixture_source_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "observation fixture must declare fixture_source_ids".to_owned())?;
    let declared_sources: Vec<&str> = source_ids
        .iter()
        .map(|source| {
            source
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .ok_or_else(|| "fixture_source_ids must contain non-empty strings".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let source_set: std::collections::HashSet<&str> = declared_sources.iter().copied().collect();
    if source_set.len() != declared_sources.len() {
        return Err("observation fixture contains duplicate source IDs".to_owned());
    }

    let events = fixture
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "observation fixture must contain observations".to_owned())?;
    let mut event_map = HashMap::with_capacity(events.len());
    for (index, event) in events.iter().enumerate() {
        if let Some(error) = observation_validator.iter_errors(event).next() {
            return Err(format!(
                "observation[{index}] schema validation failed: {} at {}",
                error,
                error.instance_path()
            ));
        }
        if event.get("fixture_only").and_then(Value::as_bool) != Some(true) {
            return Err(format!("observation[{index}] is not fixture_only"));
        }
        let id = nonempty_text(event, "observation_id")?;
        let source_id = event
            .pointer("/source/source_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{id}: missing source.source_id"))?;
        if !source_set.contains(source_id) {
            return Err(format!(
                "{id}: source {source_id} is not declared by fixture"
            ));
        }
        if event_map.insert(id.to_owned(), event).is_some() {
            return Err(format!("duplicate observation ID: {id}"));
        }
    }

    let comparison_id = nonempty_text(&comparison, "comparison_id")?;
    let pairing_id = nonempty_text(&comparison, "pairing_observation_id")?;
    let coherence_id = nonempty_text(&comparison, "coherence_observation_id")?;
    if pairing_id == coherence_id {
        return Err(format!(
            "{comparison_id}: pairing and coherence IDs must be distinct"
        ));
    }

    let pairing = event_map
        .get(pairing_id)
        .ok_or_else(|| format!("{comparison_id}: pairing event {pairing_id} does not resolve"))?;
    let coherence = event_map.get(coherence_id).ok_or_else(|| {
        format!("{comparison_id}: coherence event {coherence_id} does not resolve")
    })?;

    for (label, event, accepted_properties, accepted_scopes) in [
        (
            "pairing",
            pairing,
            vec!["pairing_energy", "pair_transfer_effective_charge"],
            vec!["local_probe", "mesoscopic"],
        ),
        (
            "coherence",
            coherence,
            vec![
                "superfluid_stiffness",
                "superfluid_density",
                "phase_correlation_length",
                "phase_correlation_time",
                "josephson_coupling_energy",
            ],
            vec!["bulk", "mesoscopic", "device"],
        ),
    ] {
        let origin = event
            .pointer("/assertion/origin")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{comparison_id}: {label} event has no assertion origin"))?;
        if origin != "observed" {
            return Err(format!(
                "{comparison_id}: {label} event must be observed, not {origin}"
            ));
        }
        let detection = event
            .pointer("/assertion/detection_status")
            .and_then(Value::as_str);
        if detection != Some("detected") {
            return Err(format!(
                "{comparison_id}: {label} event must have detected status"
            ));
        }
        let property = event
            .pointer("/assertion/property_code")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{comparison_id}: {label} event missing property_code"))?;
        if !accepted_properties.contains(&property) {
            return Err(format!(
                "{comparison_id}: {label} event property {property} is not valid for this comparison role"
            ));
        }
        let scope = event
            .pointer("/assertion/measurement_scope")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{comparison_id}: {label} event missing measurement_scope"))?;
        if !accepted_scopes.contains(&scope) {
            return Err(format!(
                "{comparison_id}: {label} event scope {scope} is invalid"
            ));
        }
        let value = event
            .pointer("/assertion/value")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("{comparison_id}: {label} event requires numeric value"))?;
        if value <= 0.0 {
            return Err(format!(
                "{comparison_id}: {label} event value must be positive"
            ));
        }
        let unit = event
            .pointer("/assertion/unit")
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{comparison_id}: {label} event missing unit"))?;
        if label == "pairing" && property == "pairing_energy" && unit != "meV" {
            return Err(format!(
                "{comparison_id}: pairing_energy must remain in meV; no energy-to-temperature conversion"
            ));
        }
    }

    let material_id = nonempty_text(&comparison, "material_id")?;
    let sample_id = nonempty_text(&comparison, "sample_id")?;
    let batch_id = nonempty_text(&comparison, "batch_id")?;
    let state_id = nonempty_text(&comparison, "state_id")?;

    for (label, event) in [("pairing", pairing), ("coherence", coherence)] {
        let observed_material = event
            .pointer("/material/material_id")
            .and_then(Value::as_str);
        let observed_sample = event.pointer("/sample/sample_id").and_then(Value::as_str);
        let observed_batch = event.pointer("/sample/batch_id").and_then(Value::as_str);
        let observed_state = event.pointer("/material/state_id").and_then(Value::as_str);
        let sample_status = event
            .pointer("/sample/identity_status")
            .and_then(Value::as_str);
        if observed_material != Some(material_id) {
            return Err(format!(
                "{comparison_id}: {label} event material identity does not match"
            ));
        }
        if observed_sample != Some(sample_id) || observed_batch != Some(batch_id) {
            return Err(format!(
                "{comparison_id}: {label} event sample/batch identity does not match"
            ));
        }
        if observed_state != Some(state_id) {
            return Err(format!(
                "{comparison_id}: {label} event material state does not match"
            ));
        }
        if sample_status != Some("matched_specimen") {
            return Err(format!(
                "{comparison_id}: {label} event lacks matched_specimen identity"
            ));
        }
    }

    let fixed_conditions = comparison
        .get("fixed_conditions")
        .ok_or_else(|| format!("{comparison_id}: missing fixed_conditions"))?;
    for (label, event) in [("pairing", pairing), ("coherence", coherence)] {
        for (fixture_field, event_path) in [
            ("pressure", "/conditions/pressure"),
            ("magnetic_field_t", "/conditions/magnetic_field_t"),
            ("field_orientation", "/conditions/field_orientation"),
            ("doping_state", "/conditions/doping_state"),
            ("strain_state", "/conditions/strain_state"),
        ] {
            let expected = fixed_conditions.get(fixture_field).ok_or_else(|| {
                format!("{comparison_id}: fixed_conditions missing {fixture_field}")
            })?;
            let actual = event
                .pointer(event_path)
                .ok_or_else(|| format!("{comparison_id}/{label}: event missing {event_path}"))?;
            let matches = if fixture_field == "pressure" {
                expected.get("semantics") == actual.get("semantics")
                    && expected.get("value_gpa") == actual.get("value_gpa")
            } else {
                expected == actual
            };
            if !matches {
                return Err(format!(
                    "{comparison_id}: {label} event does not match fixed condition {fixture_field}"
                ));
            }
        }
    }

    let pairing_value = pairing
        .pointer("/assertion/value")
        .and_then(Value::as_f64)
        .unwrap();
    let pairing_unit = pairing
        .pointer("/assertion/unit")
        .and_then(Value::as_str)
        .unwrap();
    let coherence_value = coherence
        .pointer("/assertion/value")
        .and_then(Value::as_f64)
        .unwrap();
    let coherence_unit = coherence
        .pointer("/assertion/unit")
        .and_then(Value::as_str)
        .unwrap();

    Ok(json!({
        "status": "pass",
        "contract": "SC-PAIR-COHERENCE-COMPARISON-0.1",
        "comparison_id": comparison_id,
        "pairing_event_id": pairing_id,
        "coherence_event_id": coherence_id,
        "material_id": material_id,
        "sample_id": sample_id,
        "batch_id": batch_id,
        "observations": {
            "pairing": {
                "property_code": pairing["assertion"]["property_code"],
                "value": pairing_value,
                "unit": pairing_unit,
                "measurement_scope": pairing["assertion"]["measurement_scope"],
                "measurement_temperature_k": pairing["conditions"]["measurement_temperature_k"]
            },
            "coherence": {
                "property_code": coherence["assertion"]["property_code"],
                "value": coherence_value,
                "unit": coherence_unit,
                "measurement_scope": coherence["assertion"]["measurement_scope"],
                "measurement_temperature_k": coherence["conditions"]["measurement_temperature_k"]
            }
        },
        "checks": {
            "pairing_and_coherence_are_distinct_events": true,
            "both_events_are_observed": true,
            "same_material_sample_batch_and_state": true,
            "same_declared_fixed_conditions": true,
            "local_pairing_not_promoted_to_bulk_tc": true,
            "pairing_energy_not_converted_to_temperature": true
        },
        "claim_ceiling": "Synthetic evidence-link validation only; this does not establish a pairing/coherence mechanism, Tc enhancement or room-temperature superconductivity."
    }))
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --comparison-schema <schema.json> --comparison <comparison.json> --observation-schema <schema.json> --observations <fixture.json> [--out <receipt.json>]"
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
    let mut comparison_schema_path = None;
    let mut comparison_path = None;
    let mut observation_schema_path = None;
    let mut observation_fixture_path = None;
    let mut output_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--comparison-schema"
            | "--comparison"
            | "--observation-schema"
            | "--observations"
            | "--out" => {
                if index + 1 >= args.len() {
                    usage(&args[0]);
                    process::exit(2);
                }
                let path = PathBuf::from(&args[index + 1]);
                match args[index].as_str() {
                    "--comparison-schema" => comparison_schema_path = Some(path),
                    "--comparison" => comparison_path = Some(path),
                    "--observation-schema" => observation_schema_path = Some(path),
                    "--observations" => observation_fixture_path = Some(path),
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
    let (
        Some(comparison_schema_path),
        Some(comparison_path),
        Some(observation_schema_path),
        Some(observation_fixture_path),
    ) = (
        comparison_schema_path,
        comparison_path,
        observation_schema_path,
        observation_fixture_path,
    )
    else {
        usage(&args[0]);
        process::exit(2);
    };
    let receipt = validate_comparison(
        &read(&comparison_schema_path),
        &read(&comparison_path),
        &read(&observation_schema_path),
        &read(&observation_fixture_path),
    )
    .unwrap_or_else(|error| {
        eprintln!("SC-PAIR-COHERENCE-FAIL: {error}");
        process::exit(1);
    });
    let rendered = serde_json::to_string_pretty(&receipt)
        .expect("serializing pairing/coherence receipt should not fail");
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

    fn committed_inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>, Value) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let comparison_schema = fs::read(
            root.join("data/schemas/superconductivity_pair_coherence_comparison_v0_1.schema.json"),
        )
        .expect("comparison schema must be readable");
        let comparison = fs::read(
            root.join("tools/superconductivity-discovery-harness/fixtures/pair_coherence_comparison_smoke_v0_1.json"),
        ).expect("comparison fixture must be readable");
        let observation_schema = fs::read(
            root.join("data/schemas/superconductivity_observation_event_v0_1.schema.json"),
        )
        .expect("observation schema must be readable");
        let observations: Value = serde_json::from_slice(
            &fs::read(
                root.join("tools/superconductivity-discovery-harness/fixtures/observation_events_smoke_v0_1.json"),
            ).expect("observation fixture must be readable"),
        ).expect("observation fixture must parse");
        (
            comparison_schema,
            comparison,
            observation_schema,
            observations,
        )
    }

    fn audit_mutated(comparison: Value, observations: Value) -> Result<Value, String> {
        let (comparison_schema, _, observation_schema, _) = committed_inputs();
        validate_comparison(
            &comparison_schema,
            &serde_json::to_vec(&comparison).unwrap(),
            &observation_schema,
            &serde_json::to_vec(&observations).unwrap(),
        )
    }

    fn parse_comparison() -> (Value, Value) {
        let (_, comparison, _, observations) = committed_inputs();
        (serde_json::from_slice(&comparison).unwrap(), observations)
    }

    #[test]
    fn committed_pair_coherence_comparison_passes() {
        let (comparison_schema, comparison, observation_schema, observations) = committed_inputs();
        let receipt = validate_comparison(
            &comparison_schema,
            &comparison,
            &observation_schema,
            &serde_json::to_vec(&observations).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["status"], "pass");
        assert_eq!(receipt["observations"]["pairing"]["unit"], "meV");
        assert_eq!(
            receipt["observations"]["coherence"]["property_code"],
            "superfluid_stiffness"
        );
    }

    #[test]
    fn sample_mismatch_fails_closed() {
        let (mut comparison, observations) = parse_comparison();
        comparison["sample_id"] = json!("different-sample");
        assert!(audit_mutated(comparison, observations).is_err());
    }

    #[test]
    fn batch_mismatch_fails_closed() {
        let (mut comparison, mut observations) = parse_comparison();
        observations["observations"][6]["sample"]["batch_id"] = json!("different-batch");
        assert!(audit_mutated(comparison, observations).is_err());
    }

    #[test]
    fn material_mismatch_fails_closed() {
        let (mut comparison, mut observations) = parse_comparison();
        observations["observations"][6]["material"]["material_id"] = json!("other-material");
        assert!(audit_mutated(comparison, observations).is_err());
    }

    #[test]
    fn fixed_doping_conditions_must_match() {
        let (comparison, mut observations) = parse_comparison();
        observations["observations"][6]["conditions"]["doping_state"] = json!("different doping");
        assert!(audit_mutated(comparison, observations).is_err());
    }

    #[test]
    fn prediction_cannot_be_used_as_measured_pairing() {
        let (comparison, mut observations) = parse_comparison();
        observations["observations"][5]["assertion"]["origin"] = json!("predicted");
        observations["observations"][5]["assertion"]["detection_status"] = json!("not_applicable");
        observations["observations"][5]["assertion"]["model_id"] = json!("fixture-model");
        assert!(audit_mutated(comparison, observations).is_err());
    }

    #[test]
    fn tc_is_not_accepted_as_pairing_energy() {
        let (mut comparison, mut observations) = parse_comparison();
        observations["observations"][5]["assertion"]["property_code"] = json!("Tc");
        comparison["pairing_observation_id"] = json!("fixture-local-pairing-energy");
        assert!(audit_mutated(comparison, observations).is_err());
    }

    #[test]
    fn unresolved_specimen_identity_fails_closed() {
        let (comparison, mut observations) = parse_comparison();
        observations["observations"][5]["sample"]["identity_status"] = json!("unresolved");
        assert!(audit_mutated(comparison, observations).is_err());
    }
}
