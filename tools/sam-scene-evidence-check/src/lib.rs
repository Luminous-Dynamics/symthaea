use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

pub fn validate_files(schema_path: &Path, instance_path: &Path) -> Result<(), Vec<String>> {
    let schema_text = fs::read_to_string(schema_path)
        .map_err(|error| vec![format!("read schema {}: {error}", schema_path.display())])?;
    let instance_text = fs::read_to_string(instance_path)
        .map_err(|error| vec![format!("read instance {}: {error}", instance_path.display())])?;
    let schema: Value = serde_json::from_str(&schema_text)
        .map_err(|error| vec![format!("parse schema {}: {error}", schema_path.display())])?;
    let instance: Value = serde_json::from_str(&instance_text)
        .map_err(|error| vec![format!("parse instance {}: {error}", instance_path.display())])?;
    validate_values(&schema, &instance)
}

pub fn validate_values(schema: &Value, instance: &Value) -> Result<(), Vec<String>> {
    if !jsonschema::meta::is_valid(schema) {
        return Err(vec!["schema is not valid under its declared JSON Schema meta-schema".into()]);
    }

    // The scene schema has only local $defs references. Offline mode makes an
    // unexpected remote $ref a hard error instead of a network-dependent fetch.
    let validator = jsonschema::options()
        .offline()
        .build(schema)
        .map_err(|error| vec![format!("compile schema: {error}")])?;

    let mut errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| format!("schema {}: {error}", error.instance_path()))
        .collect();

    errors.extend(validate_semantics(instance));
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_semantics(instance: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let Some(snapshot) = instance.get("snapshot") else {
        return errors;
    };

    let source_ids: BTreeSet<String> = snapshot
        .get("sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|source| source.get("source_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();

    let frame_ids: BTreeSet<String> = snapshot
        .get("reference_frames")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|frame| frame.get("frame_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();

    let track_ids: BTreeSet<String> = snapshot
        .get("object_hypotheses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|object| object.get("track_lineage_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();

    let mut duplicate_sources = BTreeSet::new();
    if let Some(sources) = snapshot.get("sources").and_then(Value::as_array) {
        for source in sources {
            if let Some(id) = source.get("source_id").and_then(Value::as_str) {
                if !duplicate_sources.insert(id.to_owned()) {
                    errors.push(format!("duplicate source_id: {id}"));
                }
            }
            check_ref(
                &mut errors,
                "source.reference_frame_id",
                source.get("reference_frame_id"),
                &frame_ids,
            );
        }
    }

    let time = snapshot.get("time");
    let event = time.and_then(|value| value.get("event_tick_ns")).and_then(Value::as_u64);
    let received = time.and_then(|value| value.get("receive_tick_ns")).and_then(Value::as_u64);
    let processed = time
        .and_then(|value| value.get("processing_tick_ns"))
        .and_then(Value::as_u64);
    if let (Some(event), Some(received), Some(processed)) = (event, received, processed) {
        if !(event <= received && received <= processed) {
            errors.push(
                "snapshot clock order must satisfy event_tick_ns <= receive_tick_ns <= processing_tick_ns"
                    .into(),
            );
        }
    }

    check_ref(
        &mut errors,
        "ego_state.reference_frame_id",
        snapshot
            .get("ego_state")
            .and_then(|value| value.get("reference_frame_id")),
        &frame_ids,
    );
    check_ref(
        &mut errors,
        "road_geometry.reference_frame_id",
        snapshot
            .get("road_geometry")
            .and_then(|value| value.get("reference_frame_id")),
        &frame_ids,
    );
    check_ref(
        &mut errors,
        "free_space.reference_frame_id",
        snapshot
            .get("free_space")
            .and_then(|value| value.get("reference_frame_id")),
        &frame_ids,
    );

    if let Some(objects) = snapshot.get("object_hypotheses").and_then(Value::as_array) {
        let mut seen_tracks = BTreeSet::new();
        for (index, object) in objects.iter().enumerate() {
            let prefix = format!("object_hypotheses[{index}]");
            if let Some(track_id) = object.get("track_lineage_id").and_then(Value::as_str) {
                if !seen_tracks.insert(track_id.to_owned()) {
                    errors.push(format!("{prefix}: duplicate active track_lineage_id {track_id}"));
                }
            }
            check_ref(
                &mut errors,
                &format!("{prefix}.reference_frame_id"),
                object.get("reference_frame_id"),
                &frame_ids,
            );
            check_source_refs(
                &mut errors,
                &format!("{prefix}.source_refs"),
                object.get("source_refs"),
                &source_ids,
            );
            validate_uncertainty(&mut errors, &prefix, object.get("uncertainty"));
            validate_class_distribution(&mut errors, &prefix, object.get("class_hypotheses"));
        }
    }

    if let Some(regions) = snapshot
        .get("free_space")
        .and_then(|value| value.get("regions"))
        .and_then(Value::as_array)
    {
        for (index, region) in regions.iter().enumerate() {
            let prefix = format!("free_space.regions[{index}]");
            check_source_refs(
                &mut errors,
                &format!("{prefix}.source_refs"),
                region.get("source_refs"),
                &source_ids,
            );
            if region.get("state").and_then(Value::as_str) == Some("observed_free")
                && region
                    .get("source_refs")
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty)
            {
                errors.push(format!(
                    "{prefix}: observed_free requires at least one source reference"
                ));
            }
        }
    }

    if let Some(proposals) = snapshot.get("vision_hdc_proposals").and_then(Value::as_array) {
        for (index, proposal) in proposals.iter().enumerate() {
            let prefix = format!("vision_hdc_proposals[{index}]");
            check_source_ref(
                &mut errors,
                &format!("{prefix}.source_id"),
                proposal.get("source_id"),
                &source_ids,
            );
            check_ref(
                &mut errors,
                &format!("{prefix}.image_frame_id"),
                proposal.get("image_frame_id"),
                &frame_ids,
            );
        }
    }

    if let Some(predictions) = snapshot.get("prediction_set_refs").and_then(Value::as_array) {
        for (index, prediction) in predictions.iter().enumerate() {
            let prefix = format!("prediction_set_refs[{index}]");
            check_track_ref(
                &mut errors,
                &format!("{prefix}.track_lineage_id"),
                prediction.get("track_lineage_id"),
                &track_ids,
            );
        }
    }

    errors
}

fn check_ref(
    errors: &mut Vec<String>,
    label: &str,
    reference: Option<&Value>,
    allowed: &BTreeSet<String>,
) {
    check_string_ref(errors, label, reference, allowed);
}

fn check_source_ref(
    errors: &mut Vec<String>,
    label: &str,
    reference: Option<&Value>,
    allowed: &BTreeSet<String>,
) {
    check_string_ref(errors, label, reference, allowed);
}

fn check_track_ref(
    errors: &mut Vec<String>,
    label: &str,
    reference: Option<&Value>,
    allowed: &BTreeSet<String>,
) {
    check_string_ref(errors, label, reference, allowed);
}

fn check_string_ref(
    errors: &mut Vec<String>,
    label: &str,
    reference: Option<&Value>,
    allowed: &BTreeSet<String>,
) {
    let Some(reference) = reference.and_then(Value::as_str) else {
        return;
    };
    if !allowed.contains(reference) {
        errors.push(format!("{label} references unknown ID {reference}"));
    }
}

fn check_source_refs(
    errors: &mut Vec<String>,
    label: &str,
    references: Option<&Value>,
    allowed: &BTreeSet<String>,
) {
    if let Some(references) = references.and_then(Value::as_array) {
        for reference in references {
            check_source_ref(errors, label, Some(reference), allowed);
        }
    }
}

fn validate_uncertainty(errors: &mut Vec<String>, prefix: &str, uncertainty: Option<&Value>) {
    let Some(uncertainty) = uncertainty else {
        return;
    };
    if uncertainty.get("representation").and_then(Value::as_str) != Some("diagonal_covariance") {
        return;
    }

    let order_len = uncertainty
        .get("state_order")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let units_len = uncertainty
        .get("units")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let diagonal = uncertainty.get("diagonal").and_then(Value::as_array);

    if order_len == 0 || order_len != units_len || diagonal.map_or(true, |values| values.len() != order_len) {
        errors.push(format!(
            "{prefix}.uncertainty: diagonal covariance dimension must match state_order and units"
        ));
        return;
    }

    if let Some(values) = diagonal {
        if values
            .iter()
            .any(|value| value.as_f64().is_none_or(|number| !number.is_finite() || number < 0.0))
        {
            errors.push(format!(
                "{prefix}.uncertainty: diagonal covariance entries must be finite and non-negative"
            ));
        }
    }
}

fn validate_class_distribution(errors: &mut Vec<String>, prefix: &str, classes: Option<&Value>) {
    let Some(classes) = classes else {
        return;
    };
    if classes.get("distribution_status").and_then(Value::as_str) != Some("calibrated_probability") {
        return;
    }
    let scores = classes.get("scores").and_then(Value::as_array);
    let Some(scores) = scores else {
        errors.push(format!("{prefix}.class_hypotheses: calibrated probability requires scores"));
        return;
    };
    let sum: f64 = scores
        .iter()
        .filter_map(|score| score.get("score").and_then(Value::as_f64))
        .sum();
    if (sum - 1.0).abs() > 1e-6 {
        errors.push(format!(
            "{prefix}.class_hypotheses: calibrated probability scores must sum to 1.0 (got {sum})"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root")
    }

    fn load_fixture() -> (Value, Value) {
        let root = repo_root();
        let schema: Value = serde_json::from_str(
            &fs::read_to_string(root.join("docs/autonomy/SAM_SCENE_EVIDENCE_SCHEMA_V0_1.json"))
                .expect("schema text"),
        )
        .expect("schema JSON");
        let fixture: Value = serde_json::from_str(
            &fs::read_to_string(
                root.join("docs/autonomy/fixtures/SAM_SCENE_EVIDENCE_CONTRACT_V0_1.json"),
            )
            .expect("fixture text"),
        )
        .expect("fixture JSON");
        (schema, fixture)
    }

    #[test]
    fn canonical_fixture_conforms_to_schema_and_semantic_contract() {
        let (schema, fixture) = load_fixture();
        validate_values(&schema, &fixture).expect("canonical fixture must conform");
    }

    #[test]
    fn rejects_any_actuator_authority() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["admission"]["actuator_authority"] = Value::String("granted".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_metric_authority_from_hdc_proposals() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["vision_hdc_proposals"][0]["metric_geometry_authority"] =
            Value::String("metric".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_unknown_source_reference() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["object_hypotheses"][0]["source_refs"][0] =
            Value::String("unregistered-source".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_covariance_dimension_mismatch() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["object_hypotheses"][0]["uncertainty"]["diagonal"]
            .as_array_mut()
            .expect("diagonal")
            .pop();
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_observed_free_region_without_source_evidence() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["free_space"]["regions"][0]["source_refs"] = Value::Array(vec![]);
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_duplicate_track_lineage() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["object_hypotheses"][1]["track_lineage_id"] =
            Value::String("track-42".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_prediction_for_unknown_track() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["prediction_set_refs"][0]["track_lineage_id"] =
            Value::String("track-missing".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_out_of_order_snapshot_clock() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["time"]["receive_tick_ns"] = Value::from(11_000_000_000u64);
        assert!(validate_values(&schema, &fixture).is_err());
    }
}
