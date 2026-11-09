use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::Value;

pub fn validate_files(schema_path: &Path, instance_path: &Path) -> Result<(), Vec<String>> {
    let schema_text = fs::read_to_string(schema_path)
        .map_err(|error| vec![format!("read schema {}: {error}", schema_path.display())])?;
    let instance_text = fs::read_to_string(instance_path).map_err(|error| {
        vec![format!(
            "read instance {}: {error}",
            instance_path.display()
        )]
    })?;
    let schema: Value = serde_json::from_str(&schema_text)
        .map_err(|error| vec![format!("parse schema {}: {error}", schema_path.display())])?;
    let instance: Value = serde_json::from_str(&instance_text).map_err(|error| {
        vec![format!(
            "parse instance {}: {error}",
            instance_path.display()
        )]
    })?;
    validate_values(&schema, &instance)
}

pub fn validate_bundle_files(
    scene_schema_path: &Path,
    scene_fixture_path: &Path,
    traffic_schema_path: &Path,
    traffic_fixture_path: &Path,
) -> Result<(), Vec<String>> {
    let scene_schema = read_json_file(scene_schema_path, "scene schema")?;
    let scene_fixture = read_json_file(scene_fixture_path, "scene fixture")?;
    let traffic_schema = read_json_file(traffic_schema_path, "traffic profile schema")?;
    let traffic_fixture = read_json_file(traffic_fixture_path, "traffic profile fixture")?;
    validate_bundle_values(
        &scene_schema,
        &scene_fixture,
        &traffic_schema,
        &traffic_fixture,
    )
}

fn read_json_file(path: &Path, label: &str) -> Result<Value, Vec<String>> {
    let text = fs::read_to_string(path)
        .map_err(|error| vec![format!("read {label} {}: {error}", path.display())])?;
    serde_json::from_str(&text)
        .map_err(|error| vec![format!("parse {label} {}: {error}", path.display())])
}

pub fn validate_bundle_values(
    scene_schema: &Value,
    scene_fixture: &Value,
    traffic_schema: &Value,
    traffic_fixture: &Value,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if let Err(mut validation_errors) = validate_values(scene_schema, scene_fixture) {
        errors.append(&mut validation_errors);
    }
    if let Err(mut validation_errors) = validate_traffic_values(traffic_schema, traffic_fixture) {
        errors.append(&mut validation_errors);
    }

    let scene_profile_id = scene_fixture
        .get("snapshot")
        .and_then(|snapshot| snapshot.get("traffic_convention_profile_id"))
        .and_then(Value::as_str);
    let known_profile = traffic_fixture
        .get("profiles")
        .and_then(Value::as_array)
        .is_some_and(|profiles| {
            profiles.iter().any(|profile| {
                profile.get("profile_id").and_then(Value::as_str) == scene_profile_id
            })
        });
    if !known_profile {
        errors.push(format!(
            "scene snapshot traffic_convention_profile_id {:?} does not resolve in the traffic profile registry",
            scene_profile_id
        ));
    } else {
        let profile = traffic_fixture
            .get("profiles")
            .and_then(Value::as_array)
            .and_then(|profiles| {
                profiles.iter().find(|profile| {
                    profile.get("profile_id").and_then(Value::as_str) == scene_profile_id
                })
            });
        let profile_frame_id = profile
            .and_then(|profile| profile.get("travel_frame_id"))
            .and_then(Value::as_str);
        let snapshot = scene_fixture.get("snapshot");
        let frame = snapshot
            .and_then(|snapshot| snapshot.get("reference_frames"))
            .and_then(Value::as_array)
            .and_then(|frames| {
                frames
                    .iter()
                    .find(|frame| frame.get("frame_id").and_then(Value::as_str) == profile_frame_id)
            });
        if frame.is_none() {
            errors.push(format!(
                "traffic profile {:?} travel_frame_id {:?} does not resolve in the scene reference-frame registry",
                scene_profile_id, profile_frame_id
            ));
        } else if frame
            .and_then(|frame| frame.get("kind"))
            .and_then(Value::as_str)
            != Some("local_road")
        {
            errors.push(format!(
                "traffic profile {:?} travel_frame_id {:?} is not a local_road frame",
                scene_profile_id, profile_frame_id
            ));
        }
        let ego_frame = snapshot
            .and_then(|snapshot| snapshot.get("ego_state"))
            .and_then(|ego| ego.get("reference_frame_id"))
            .and_then(Value::as_str);
        let road_frame = snapshot
            .and_then(|snapshot| snapshot.get("road_geometry"))
            .and_then(|road| road.get("reference_frame_id"))
            .and_then(Value::as_str);
        if profile_frame_id != ego_frame || profile_frame_id != road_frame {
            errors.push(format!(
                "traffic profile {:?} travel frame {:?} must match both ego frame {:?} and road frame {:?}",
                scene_profile_id, profile_frame_id, ego_frame, road_frame
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn validate_values(schema: &Value, instance: &Value) -> Result<(), Vec<String>> {
    if !jsonschema::meta::is_valid(schema) {
        return Err(vec![
            "schema is not valid under its declared JSON Schema meta-schema".into(),
        ]);
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

pub fn validate_traffic_files(schema_path: &Path, fixture_path: &Path) -> Result<(), Vec<String>> {
    let schema_text = fs::read_to_string(schema_path).map_err(|error| {
        vec![format!(
            "read traffic schema {}: {error}",
            schema_path.display()
        )]
    })?;
    let fixture_text = fs::read_to_string(fixture_path).map_err(|error| {
        vec![format!(
            "read traffic fixture {}: {error}",
            fixture_path.display()
        )]
    })?;
    let schema: Value = serde_json::from_str(&schema_text).map_err(|error| {
        vec![format!(
            "parse traffic schema {}: {error}",
            schema_path.display()
        )]
    })?;
    let fixture: Value = serde_json::from_str(&fixture_text).map_err(|error| {
        vec![format!(
            "parse traffic fixture {}: {error}",
            fixture_path.display()
        )]
    })?;
    validate_traffic_values(&schema, &fixture)
}

pub fn validate_traffic_values(schema: &Value, fixture: &Value) -> Result<(), Vec<String>> {
    if !jsonschema::meta::is_valid(schema) {
        return Err(vec![
            "traffic schema is not valid under its declared JSON Schema meta-schema".into(),
        ]);
    }
    let validator = jsonschema::options()
        .offline()
        .build(schema)
        .map_err(|error| vec![format!("compile traffic schema: {error}")])?;
    let mut errors: Vec<String> = validator
        .iter_errors(fixture)
        .map(|error| format!("traffic schema {}: {error}", error.instance_path()))
        .collect();
    errors.extend(validate_traffic_semantics(fixture));
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_traffic_semantics(fixture: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let profiles = fixture.get("profiles").and_then(Value::as_array);
    let Some(profiles) = profiles else {
        return errors;
    };
    let evidence_records = fixture
        .get("evidence_records")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut evidence_index = std::collections::BTreeMap::new();
    for (index, evidence) in evidence_records.iter().enumerate() {
        let Some(id) = evidence.get("evidence_id").and_then(Value::as_str) else {
            errors.push(format!("evidence_records[{index}]: missing evidence_id"));
            continue;
        };
        if evidence_index.insert(id.to_owned(), evidence).is_some() {
            errors.push(format!("duplicate traffic evidence_id: {id}"));
        }
    }

    let mut profile_index = std::collections::BTreeMap::new();
    for (index, profile) in profiles.iter().enumerate() {
        let Some(id) = profile.get("profile_id").and_then(Value::as_str) else {
            errors.push(format!("profiles[{index}]: missing profile_id"));
            continue;
        };
        if profile_index.insert(id.to_owned(), profile).is_some() {
            errors.push(format!("duplicate traffic profile_id: {id}"));
        }
        let traffic_side = profile.get("traffic_side").and_then(Value::as_str);
        let keep = profile.get("keep_side_sign").and_then(Value::as_i64);
        let kerb = profile.get("kerbside_sign").and_then(Value::as_i64);
        let median = profile.get("median_side_sign").and_then(Value::as_i64);
        let expected = match traffic_side {
            Some("left") => Some((1, 1, -1)),
            Some("right") => Some((-1, -1, 1)),
            _ => None,
        };
        if let Some((expected_keep, expected_kerb, expected_median)) = expected
            && (keep != Some(expected_keep)
                || kerb != Some(expected_kerb)
                || median != Some(expected_median))
        {
            errors.push(format!(
                "profiles[{index}]: traffic-side signs do not match the declared travel-frame convention"
            ));
        }
        let applicability = profile.get("applicability").and_then(Value::as_str);
        let passing_rule_id = profile
            .get("passing_rule_profile_id")
            .and_then(Value::as_str);
        let passing_rule_version = profile
            .get("passing_rule_profile_version")
            .and_then(Value::as_str);
        if traffic_side == Some("unknown") {
            if profile.get("passing_side_sign") != Some(&Value::Null)
                || applicability != Some("unknown")
                || profile.get("passing_rule_profile_id") != Some(&Value::Null)
                || profile.get("passing_rule_profile_version") != Some(&Value::Null)
            {
                errors.push(format!(
                    "profiles[{index}]: unknown traffic side must have no passing-side or passing-rule authority and unknown applicability"
                ));
            }
        } else if passing_rule_id.is_none_or(str::is_empty)
            || passing_rule_version.is_none_or(str::is_empty)
        {
            errors.push(format!(
                "profiles[{index}]: known traffic side requires an explicit passing-rule profile ID and version"
            ));
        }

        if applicability == Some("verified_for_declared_context") {
            let evidence_refs = profile
                .get("verification_evidence_refs")
                .and_then(Value::as_array);
            if evidence_refs.is_none_or(Vec::is_empty)
                || profile
                    .get("verification_method_id")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                || profile
                    .get("verification_record_id")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                errors.push(format!(
                    "profiles[{index}]: verified applicability requires evidence references, method ID and verification record ID"
                ));
            }
            let from = profile
                .get("verification_valid_from")
                .and_then(Value::as_str);
            let until = profile
                .get("verification_valid_until")
                .and_then(Value::as_str);
            if from.is_none_or(|date| date.len() != 10)
                || until.is_none_or(|date| date.len() != 10)
                || from.zip(until).is_some_and(|(from, until)| from > until)
            {
                errors.push(format!(
                    "profiles[{index}]: verified applicability requires an ordered YYYY-MM-DD validity interval"
                ));
            }

            let mut has_rule_authority_evidence = false;
            if let Some(references) = evidence_refs {
                for reference in references {
                    let evidence_id = reference.as_str().unwrap_or_default();
                    let Some(evidence) = evidence_index.get(evidence_id).copied() else {
                        errors.push(format!(
                            "profiles[{index}]: verification evidence reference {evidence_id:?} does not resolve in evidence_records"
                        ));
                        continue;
                    };
                    if evidence.get("jurisdiction_profile_id")
                        != profile.get("jurisdiction_profile_id")
                        || evidence.get("jurisdiction_profile_version")
                            != profile.get("jurisdiction_profile_version")
                    {
                        errors.push(format!(
                            "profiles[{index}]: evidence {evidence_id:?} is for a different jurisdiction profile/version"
                        ));
                    }
                    let evidence_kind = evidence.get("evidence_kind").and_then(Value::as_str);
                    if matches!(evidence_kind, Some("traffic_law" | "road_authority")) {
                        has_rule_authority_evidence = true;
                    }
                    let evidence_from = evidence.get("effective_from").and_then(Value::as_str);
                    let evidence_until = evidence.get("effective_until").and_then(Value::as_str);
                    if from
                        .zip(evidence_from)
                        .is_some_and(|(profile_from, source_from)| profile_from < source_from)
                        || until
                            .zip(evidence_until)
                            .is_some_and(|(profile_until, source_until)| {
                                profile_until > source_until
                            })
                    {
                        errors.push(format!(
                            "profiles[{index}]: verification validity exceeds evidence {evidence_id:?} effective dates"
                        ));
                    }
                }
            }
            if !has_rule_authority_evidence {
                errors.push(format!(
                    "profiles[{index}]: verified applicability requires at least one traffic_law or road_authority evidence record"
                ));
            }
        }
    }

    // Steering-wheel position is independent of traffic side. Any profiles
    // sharing a traffic side must preserve the road/lane sign convention.
    for left in profiles {
        for right in profiles {
            if left.get("profile_id") == right.get("profile_id")
                || left.get("traffic_side") != right.get("traffic_side")
            {
                continue;
            }
            for field in ["keep_side_sign", "kerbside_sign", "median_side_sign"] {
                if left.get(field) != right.get(field) {
                    errors.push(format!(
                        "profiles with equal traffic_side disagree on {field}; steering-wheel position must not alter road semantics"
                    ));
                }
            }
        }
    }

    let Some(scenes) = fixture.get("paired_scenes").and_then(Value::as_array) else {
        return errors;
    };
    let mut pairs: std::collections::BTreeMap<String, Vec<&Value>> =
        std::collections::BTreeMap::new();
    for (index, scene) in scenes.iter().enumerate() {
        let Some(pair_id) = scene.get("pair_id").and_then(Value::as_str) else {
            errors.push(format!("paired_scenes[{index}]: missing pair_id"));
            continue;
        };
        pairs.entry(pair_id.to_owned()).or_default().push(scene);
        let profile_id = scene
            .get("traffic_convention_profile_id")
            .and_then(Value::as_str);
        if profile_id.is_none_or(|id| !profile_index.contains_key(id)) {
            errors.push(format!(
                "paired_scenes[{index}]: traffic convention profile reference is unresolved"
            ));
            continue;
        }
        let profile = profile_index.get(profile_id.unwrap()).unwrap();
        if scene.get("declared_passing_side_sign") != profile.get("passing_side_sign") {
            errors.push(format!(
                "paired_scenes[{index}]: declared passing side conflicts with the versioned rule profile"
            ));
        }
        if scene.get("map_traffic_convention_profile_id")
            != scene.get("traffic_convention_profile_id")
        {
            errors.push(format!(
                "paired_scenes[{index}]: map and scene traffic convention profiles disagree"
            ));
        }
        if scene.get("travel_frame_id") != profile.get("travel_frame_id") {
            errors.push(format!(
                "paired_scenes[{index}]: scene travel frame does not match the traffic profile frame"
            ));
        }
        let keep_sign = profile.get("keep_side_sign");
        let median_sign = profile.get("median_side_sign");
        let mut ego_lanes = 0usize;
        let mut oncoming_lanes = 0usize;
        if let Some(lanes) = scene.get("lanes").and_then(Value::as_array) {
            for lane in lanes {
                match lane.get("role").and_then(Value::as_str) {
                    Some("ego_travel_lane") => {
                        ego_lanes += 1;
                        if lane.get("relative_lateral_sign") != keep_sign {
                            errors.push(format!(
                                "paired_scenes[{index}]: ego lane side conflicts with the traffic keep-side sign"
                            ));
                        }
                    }
                    Some("oncoming_lane") => {
                        oncoming_lanes += 1;
                        if lane.get("relative_lateral_sign") != median_sign {
                            errors.push(format!(
                                "paired_scenes[{index}]: oncoming lane side conflicts with the median-side sign"
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
        if ego_lanes != 1 || oncoming_lanes != 1 {
            errors.push(format!(
                "paired_scenes[{index}]: expected exactly one ego travel lane and one oncoming lane"
            ));
        }
    }

    for (pair_id, pair) in pairs {
        if pair.len() != 2 {
            errors.push(format!(
                "pair {pair_id}: expected exactly two mirrored scenes, got {}",
                pair.len()
            ));
            continue;
        }
        let (Some(a_id), Some(b_id)) = (
            pair[0]
                .get("traffic_convention_profile_id")
                .and_then(Value::as_str),
            pair[1]
                .get("traffic_convention_profile_id")
                .and_then(Value::as_str),
        ) else {
            errors.push(format!("pair {pair_id}: missing traffic profile reference"));
            continue;
        };
        let (Some(a_profile), Some(b_profile)) = (profile_index.get(a_id), profile_index.get(b_id))
        else {
            continue;
        };
        let a_side = a_profile.get("traffic_side").and_then(Value::as_str);
        let b_side = b_profile.get("traffic_side").and_then(Value::as_str);
        if !matches!(
            (a_side, b_side),
            (Some("left"), Some("right")) | (Some("right"), Some("left"))
        ) {
            errors.push(format!(
                "pair {pair_id}: traffic_side must flip left <-> right"
            ));
        }
        if pair[0].get("scenario_family") != pair[1].get("scenario_family") {
            errors.push(format!("pair {pair_id}: scenario families do not match"));
        }
        validate_mirrored_pair(&mut errors, &pair_id, pair[0], pair[1]);
    }
    errors
}

fn approx_eq(actual: Option<f64>, expected: Option<f64>) -> bool {
    match (actual, expected) {
        (Some(actual), Some(expected)) => (actual - expected).abs() <= 1e-9,
        _ => false,
    }
}

fn number_at(value: Option<&Value>, index: usize) -> Option<f64> {
    value?.as_array()?.get(index)?.as_f64()
}

fn validate_mirrored_pair(errors: &mut Vec<String>, pair_id: &str, a: &Value, b: &Value) {
    let a_state = a.get("ego_state");
    let b_state = b.get("ego_state");
    let fields = [
        ("position_m", 0usize, 1i8),
        ("position_m", 1usize, -1i8),
        ("position_m", 2usize, 1i8),
        ("velocity_mps", 0usize, 1i8),
        ("velocity_mps", 1usize, -1i8),
        ("velocity_mps", 2usize, 1i8),
    ];
    for (field, index, sign) in fields {
        let av = number_at(a_state.and_then(|state| state.get(field)), index);
        let bv = number_at(b_state.and_then(|state| state.get(field)), index);
        let expected = av.map(|value| value * f64::from(sign));
        if !approx_eq(bv, expected) {
            errors.push(format!(
                "pair {pair_id}: {field}[{index}] violates mirror sign/magnitude invariant"
            ));
        }
    }
    for (field, sign) in [
        ("heading_rad", -1.0),
        ("yaw_rate_rad_s", -1.0),
        ("lateral_offset_m", -1.0),
        ("curvature_per_m", -1.0),
        ("steering_angle_rad", -1.0),
    ] {
        let av = a_state
            .and_then(|state| state.get(field))
            .and_then(Value::as_f64);
        let bv = b_state
            .and_then(|state| state.get(field))
            .and_then(Value::as_f64);
        if !approx_eq(bv, av.map(|value| value * sign)) {
            errors.push(format!("pair {pair_id}: {field} violates mirror invariant"));
        }
    }
    let a_command = a.get("candidate_command");
    let b_command = b.get("candidate_command");
    for (field, sign) in [("steering", -1.0), ("throttle", 1.0), ("brake", 1.0)] {
        let av = a_command
            .and_then(|command| command.get(field))
            .and_then(Value::as_f64);
        let bv = b_command
            .and_then(|command| command.get(field))
            .and_then(Value::as_f64);
        if !approx_eq(bv, av.map(|value| value * sign)) {
            errors.push(format!(
                "pair {pair_id}: command.{field} violates mirror invariant"
            ));
        }
    }
    let a_lanes = a.get("lanes").and_then(Value::as_array);
    let b_lanes = b.get("lanes").and_then(Value::as_array);
    if let (Some(a_lanes), Some(b_lanes)) = (a_lanes, b_lanes) {
        if a_lanes.len() != b_lanes.len() {
            errors.push(format!("pair {pair_id}: lane count changed"));
        } else {
            for (index, (a_lane, b_lane)) in a_lanes.iter().zip(b_lanes).enumerate() {
                if a_lane.get("role") != b_lane.get("role")
                    || a_lane.get("lane_id") != b_lane.get("lane_id")
                {
                    errors.push(format!(
                        "pair {pair_id}: lane role / identity changed at index {index}"
                    ));
                }
                let field = "center_lateral_m";
                let av = a_lane.get(field).and_then(Value::as_f64);
                let bv = b_lane.get(field).and_then(Value::as_f64);
                if !approx_eq(bv, av.map(|value| -value)) {
                    errors.push(format!(
                        "pair {pair_id}: lane {index} field {field} did not reflect"
                    ));
                }
                if a_lane.get("relative_lateral_sign") == b_lane.get("relative_lateral_sign") {
                    errors.push(format!(
                        "pair {pair_id}: lane {index} relative lateral sign did not flip"
                    ));
                }
            }
        }
    } else {
        errors.push(format!(
            "pair {pair_id}: lanes must be present in both scenes"
        ));
    }

    let a_pass = a.get("declared_passing_side_sign").and_then(Value::as_i64);
    let b_pass = b.get("declared_passing_side_sign").and_then(Value::as_i64);
    if a_pass.zip(b_pass).is_none_or(|(a, b)| a != -b) {
        errors.push(format!("pair {pair_id}: passing-side sign did not flip"));
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

    let source_roles: BTreeMap<String, String> = snapshot
        .get("sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|source| {
            Some((
                source.get("source_id")?.as_str()?.to_owned(),
                source.get("source_role")?.as_str()?.to_owned(),
            ))
        })
        .collect();

    let source_records: BTreeMap<String, Value> = snapshot
        .get("sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|source| {
            Some((
                source.get("source_id")?.as_str()?.to_owned(),
                source.clone(),
            ))
        })
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
            if let Some(id) = source.get("source_id").and_then(Value::as_str)
                && !duplicate_sources.insert(id.to_owned())
            {
                errors.push(format!("duplicate source_id: {id}"));
            }
            validate_source_role(&mut errors, source);
            check_ref(
                &mut errors,
                "source.reference_frame_id",
                source.get("reference_frame_id"),
                &frame_ids,
            );
        }
    }

    let time = snapshot.get("time");
    let event = time
        .and_then(|value| value.get("event_tick_ns"))
        .and_then(Value::as_u64);
    let received = time
        .and_then(|value| value.get("receive_tick_ns"))
        .and_then(Value::as_u64);
    let processed = time
        .and_then(|value| value.get("processing_tick_ns"))
        .and_then(Value::as_u64);
    if let (Some(event), Some(received), Some(processed)) = (event, received, processed)
        && !(event <= received && received <= processed)
    {
        errors.push(
            "snapshot clock order must satisfy event_tick_ns <= receive_tick_ns <= processing_tick_ns"
                .into(),
        );
    }

    check_ref(
        &mut errors,
        "ego_state.reference_frame_id",
        snapshot
            .get("ego_state")
            .and_then(|value| value.get("reference_frame_id")),
        &frame_ids,
    );
    check_source_refs(
        &mut errors,
        "ego_state.source_refs",
        snapshot
            .get("ego_state")
            .and_then(|value| value.get("source_refs")),
        &source_ids,
    );
    check_source_role_refs(
        &mut errors,
        "ego_state.source_refs",
        snapshot
            .get("ego_state")
            .and_then(|value| value.get("source_refs")),
        &source_roles,
        &["local_sensor_observation", "synthetic_sensor_observation"],
    );
    validate_uncertainty(
        &mut errors,
        "ego_state",
        snapshot
            .get("ego_state")
            .and_then(|state| state.get("uncertainty")),
    );
    check_ref(
        &mut errors,
        "road_geometry.reference_frame_id",
        snapshot
            .get("road_geometry")
            .and_then(|value| value.get("reference_frame_id")),
        &frame_ids,
    );
    if let Some(lanes) = snapshot
        .get("road_geometry")
        .and_then(|value| value.get("lane_hypotheses"))
        .and_then(Value::as_array)
    {
        for (index, lane) in lanes.iter().enumerate() {
            check_source_refs(
                &mut errors,
                &format!("road_geometry.lane_hypotheses[{index}].source_refs"),
                lane.get("source_refs"),
                &source_ids,
            );
            check_source_role_refs(
                &mut errors,
                &format!("road_geometry.lane_hypotheses[{index}].source_refs"),
                lane.get("source_refs"),
                &source_roles,
                &[
                    "local_sensor_observation",
                    "synthetic_sensor_observation",
                    "map_prior",
                ],
            );
        }
    }
    check_ref(
        &mut errors,
        "free_space.reference_frame_id",
        snapshot
            .get("free_space")
            .and_then(|value| value.get("reference_frame_id")),
        &frame_ids,
    );

    if let Some(objects) = snapshot.get("object_hypotheses").and_then(Value::as_array) {
        let mut seen_hypotheses = BTreeSet::new();
        for (index, object) in objects.iter().enumerate() {
            let prefix = format!("object_hypotheses[{index}]");
            if let Some(hypothesis_id) = object.get("hypothesis_id").and_then(Value::as_str)
                && !seen_hypotheses.insert(hypothesis_id.to_owned())
            {
                errors.push(format!("{prefix}: duplicate hypothesis_id {hypothesis_id}"));
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
            check_source_role_refs(
                &mut errors,
                &format!("{prefix}.source_refs"),
                object.get("source_refs"),
                &source_roles,
                &["local_sensor_observation", "synthetic_sensor_observation"],
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
            check_source_role_refs(
                &mut errors,
                &format!("{prefix}.source_refs"),
                region.get("source_refs"),
                &source_roles,
                &["local_sensor_observation", "synthetic_sensor_observation"],
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

    if let Some(proposals) = snapshot
        .get("vision_hdc_proposals")
        .and_then(Value::as_array)
    {
        for (index, proposal) in proposals.iter().enumerate() {
            let prefix = format!("vision_hdc_proposals[{index}]");
            check_source_ref(
                &mut errors,
                &format!("{prefix}.source_id"),
                proposal.get("source_id"),
                &source_ids,
            );
            check_source_role_ref(
                &mut errors,
                &format!("{prefix}.source_id"),
                proposal.get("source_id"),
                &source_roles,
                &["local_sensor_observation", "synthetic_sensor_observation"],
            );
            check_ref(
                &mut errors,
                &format!("{prefix}.image_frame_id"),
                proposal.get("image_frame_id"),
                &frame_ids,
            );
        }
    }

    if let Some(predictions) = snapshot
        .get("prediction_set_refs")
        .and_then(Value::as_array)
    {
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

    check_source_refs(
        &mut errors,
        "cooperative_advisory_refs",
        snapshot.get("cooperative_advisory_refs"),
        &source_ids,
    );
    check_source_role_refs(
        &mut errors,
        "cooperative_advisory_refs",
        snapshot.get("cooperative_advisory_refs"),
        &source_roles,
        &["remote_cooperative_claim"],
    );
    check_source_refs(
        &mut errors,
        "simulator_truth_refs",
        snapshot.get("simulator_truth_refs"),
        &source_ids,
    );
    check_source_role_refs(
        &mut errors,
        "simulator_truth_refs",
        snapshot.get("simulator_truth_refs"),
        &source_roles,
        &["simulator_oracle"],
    );
    validate_simulator_truth_actor_states(&mut errors, snapshot, &source_records, &frame_ids);
    validate_follow_target_context(&mut errors, snapshot, &source_records, &frame_ids);

    errors
}

fn longitudinal_sigma_upper_bound(uncertainty: Option<&Value>) -> Option<f64> {
    let uncertainty = uncertainty?;
    if uncertainty.get("representation").and_then(Value::as_str)
        != Some("diagonal_covariance")
    {
        return None;
    }
    let state_order = uncertainty.get("state_order")?.as_array()?;
    let x_index = state_order
        .iter()
        .position(|field| field.as_str() == Some("x_m"))?;
    let variance = uncertainty
        .get("diagonal")?
        .as_array()?
        .get(x_index)?
        .as_f64()?;
    if !variance.is_finite() || variance < 0.0 {
        return None;
    }
    Some(variance.sqrt())
}

fn validate_simulator_truth_actor_states(
    errors: &mut Vec<String>,
    snapshot: &Value,
    source_records: &BTreeMap<String, Value>,
    frame_ids: &BTreeSet<String>,
) {
    let Some(actors) = snapshot
        .get("simulator_truth_actor_states")
        .and_then(Value::as_array)
    else {
        errors.push("simulator_truth_actor_states: required array is missing".into());
        return;
    };
    let scenario_id = snapshot.get("scenario_id").and_then(Value::as_str);
    let truth_refs: BTreeSet<&str> = snapshot
        .get("simulator_truth_refs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut seen_actor_ids = BTreeSet::new();

    for (index, actor) in actors.iter().enumerate() {
        let prefix = format!("simulator_truth_actor_states[{index}]");
        let actor_id = actor.get("actor_id").and_then(Value::as_str);
        let actor_scenario = actor.get("scenario_id").and_then(Value::as_str);
        let frame_id = actor.get("reference_frame_id").and_then(Value::as_str);
        let source_ref = actor.get("source_ref").and_then(Value::as_str);
        let acquisition_tick = actor.get("acquisition_tick_ns").and_then(Value::as_u64);

        if let Some(actor_id) = actor_id {
            if !seen_actor_ids.insert(actor_id.to_owned()) {
                errors.push(format!("{prefix}: duplicate actor_id {actor_id}"));
            }
        }
        if actor_scenario != scenario_id {
            errors.push(format!(
                "{prefix}: scenario_id must match the containing scene"
            ));
        }
        if frame_id.is_none_or(|frame| !frame_ids.contains(frame)) {
            errors.push(format!(
                "{prefix}: reference_frame_id must resolve in the scene"
            ));
        }
        let Some(source_ref) = source_ref else {
            errors.push(format!("{prefix}: source_ref is required"));
            continue;
        };
        if !truth_refs.contains(source_ref) {
            errors.push(format!(
                "{prefix}: source_ref must also appear in simulator_truth_refs"
            ));
        }
        let Some(source) = source_records.get(source_ref) else {
            errors.push(format!(
                    "{prefix}: unknown simulator-oracle source {source_ref}"
                ));
            continue;
        };
        if source.get("source_role").and_then(Value::as_str) != Some("simulator_oracle") {
            errors.push(format!(
                "{prefix}: source_ref must have simulator_oracle role"
            ));
        }
        if source.get("modality").and_then(Value::as_str) != Some("synthetic")
            || source.get("evidence_kind").and_then(Value::as_str) != Some("synthetic_observation")
        {
            errors.push(format!(
                "{prefix}: source must be a synthetic oracle observation"
            ));
        }
        if source
            .get("currentness")
            .and_then(Value::as_str)
            .is_none_or(|value| !matches!(value, "current" | "current_for_fixture"))
        {
            errors.push(format!(
                "{prefix}: stale or unknown oracle source cannot define truth state"
            ));
        }
        if source.get("reference_frame_id").and_then(Value::as_str) != frame_id {
            errors.push(format!(
                "{prefix}: actor and oracle source frames must match"
            ));
        }
        if source.get("acquisition_tick_ns").and_then(Value::as_u64) != acquisition_tick {
            errors.push(format!(
                "{prefix}: actor acquisition tick must match oracle source"
            ));
        }
        for field in ["position_m", "velocity_mps"] {
            let values = actor.get(field).and_then(Value::as_array);
            if values.is_none_or(|values| {
                values.len() != 3
                    || values
                        .iter()
                        .any(|value| value.as_f64().is_none_or(|number| !number.is_finite()))
            }) {
                errors.push(format!(
                    "{prefix}.{field}: expected a finite metric 3-vector"
                ));
            }
        }
    }
}

fn validate_follow_target_context(
    errors: &mut Vec<String>,
    snapshot: &Value,
    source_records: &BTreeMap<String, Value>,
    frame_ids: &BTreeSet<String>,
) {
    let Some(context) = snapshot.get("follow_target_context") else {
        errors.push("follow_target_context: missing typed follow evidence context".into());
        return;
    };
    let profile = context
        .get("consumer_profile")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let status = context
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    if status != "available" {
        let unresolved_fields = [
            "evidence_source_ref",
            "scenario_id",
            "actor_id",
            "coordinate_frame_id",
            "observation_tick_ns",
            "max_age_ns",
            "gap_m",
            "relative_speed_mps",
            "distance_uncertainty_m",
            "target_reference_kind",
            "target_reference_id",
        ];
        for field in unresolved_fields {
            if context.get(field).is_some_and(|value| !value.is_null()) {
                errors.push(format!(
                    "follow_target_context: {status} status cannot retain {field}; no oracle or remote fallback is permitted"
                ));
            }
        }
        return;
    }

    for field in [
        "scenario_id",
        "actor_id",
        "coordinate_frame_id",
        "target_reference_kind",
        "target_reference_id",
    ] {
        if context
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            errors.push(format!(
                "follow_target_context: available target requires a non-empty {field}"
            ));
        }
    }

    let scenario = context.get("scenario_id").and_then(Value::as_str);
    let snapshot_scenario = snapshot.get("scenario_id").and_then(Value::as_str);
    if scenario != snapshot_scenario {
        errors
            .push("follow_target_context: scenario_id must match the exact scene scenario".into());
    }

    let source_id = context.get("evidence_source_ref").and_then(Value::as_str);
    let Some(source_id) = source_id else {
        errors.push("follow_target_context: available target requires evidence_source_ref".into());
        return;
    };
    let Some(source) = source_records.get(source_id) else {
        errors.push(format!(
            "follow_target_context: unknown evidence source {source_id}"
        ));
        return;
    };

    let role = source
        .get("source_role")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let reference_kind = context
        .get("target_reference_kind")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let reference_id = context
        .get("target_reference_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let expected_role = match profile {
        "simulation_benchmark" => "simulator_oracle",
        "local_perception" => {
            if !matches!(
                role,
                "local_sensor_observation" | "synthetic_sensor_observation"
            ) {
                errors.push(format!(
                    "follow_target_context: local_perception requires a local sensor observation, not {role}"
                ));
            }
            role
        }
        "deployment" => "local_sensor_observation",
        _ => {
            errors.push(format!(
                "follow_target_context: unsupported consumer profile {profile}"
            ));
            return;
        }
    };
    if profile != "local_perception" && role != expected_role {
        errors.push(format!(
            "follow_target_context: {profile} requires source role {expected_role}, not {role}; remote cooperative claims never substitute for a local lead track"
        ));
    }

    let source_modality = source
        .get("modality")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let evidence_kind = source
        .get("evidence_kind")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let currentness = source
        .get("currentness")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let actor_sensor_modality = matches!(
        source_modality,
        "rgb_camera" | "stereo_camera" | "lidar" | "radar" | "ultrasonic"
    );

    match profile {
        "simulation_benchmark" => {
            if source_modality != "synthetic" || evidence_kind != "synthetic_observation" {
                errors.push(
                    "follow_target_context: simulation benchmark oracle must be a synthetic observation"
                        .into(),
                );
            }
            if !matches!(currentness, "current" | "current_for_fixture") {
                errors.push(
                    "follow_target_context: stale or unknown simulator oracle cannot drive Follow"
                        .into(),
                );
            }
            if reference_kind != "simulator_truth_actor"
                || reference_id
                    != context
                        .get("actor_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
            {
                errors.push(
                    "follow_target_context: simulation benchmark must link to the exact simulator truth actor"
                        .into(),
                );
            }
        }
        "local_perception" => {
            if !actor_sensor_modality {
                errors.push(
                    "follow_target_context: local perception Follow requires camera/LiDAR/radar/ultrasonic actor evidence, not proprioceptive or remote modalities"
                        .into(),
                );
            }
            let admissible = match role {
                "local_sensor_observation" => {
                    evidence_kind == "measurement" && currentness == "current"
                }
                "synthetic_sensor_observation" => {
                    evidence_kind == "synthetic_observation" && currentness == "current_for_fixture"
                }
                _ => false,
            };
            if !admissible {
                errors.push(
                    "follow_target_context: local perception source must be a current local measurement or explicitly synthetic fixture observation"
                        .into(),
                );
            }
            if reference_kind != "tracked_object_hypothesis" {
                errors.push(
                    "follow_target_context: local perception must link to a tracked object hypothesis"
                        .into(),
                );
            }
        }
        "deployment" => {
            if !actor_sensor_modality || evidence_kind != "measurement" || currentness != "current"
            {
                errors.push(
                    "follow_target_context: deployment Follow requires a current camera/LiDAR/radar/ultrasonic local sensor measurement"
                        .into(),
                );
            }
            if reference_kind != "tracked_object_hypothesis" {
                errors.push(
                    "follow_target_context: deployment must link to a tracked object hypothesis, not simulator truth"
                        .into(),
                );
            }
        }
        _ => {}
    }

    let context_frame = context.get("coordinate_frame_id").and_then(Value::as_str);
    if context_frame.is_none_or(|frame| !frame_ids.contains(frame)) {
        errors.push("follow_target_context: coordinate_frame_id must resolve in the scene".into());
    }
    let source_frame = source.get("reference_frame_id").and_then(Value::as_str);
    if source_frame != context_frame {
        errors.push(
            "follow_target_context: evidence source frame must match the declared target frame"
                .into(),
        );
    }
    let road_frame_is_local = snapshot
        .get("reference_frames")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|frame| {
            frame.get("frame_id").and_then(Value::as_str) == context_frame
                && frame.get("kind").and_then(Value::as_str) == Some("local_road")
        });
    if !road_frame_is_local {
        errors.push(
            "follow_target_context: v0.1 gap and relative speed require the declared local_road x-forward frame"
                .into(),
        );
    }

    let event_tick = snapshot
        .get("time")
        .and_then(|time| time.get("event_tick_ns"))
        .and_then(Value::as_u64);
    let source_tick = source.get("acquisition_tick_ns").and_then(Value::as_u64);
    let observation_tick = context.get("observation_tick_ns").and_then(Value::as_u64);
    let max_age = context.get("max_age_ns").and_then(Value::as_u64);
    let source_clock = source.get("clock_domain_id").and_then(Value::as_str);
    let scene_clock = snapshot
        .get("time")
        .and_then(|time| time.get("clock_domain_id"))
        .and_then(Value::as_str);
    let source_transform = source.get("clock_transform_id").and_then(Value::as_str);
    let scene_transform = snapshot
        .get("time")
        .and_then(|time| time.get("clock_transform_id"))
        .and_then(Value::as_str);

    if source_tick != observation_tick
        || source_clock != scene_clock
        || source_transform != scene_transform
    {
        errors.push(
            "follow_target_context: observation time must be normalized under the exact scene clock/transform"
                .into(),
        );
    }
    match (event_tick, observation_tick, max_age) {
        (Some(now), Some(observed), Some(maximum))
            if observed <= now && now - observed <= maximum => {}
        _ => errors.push(
            "follow_target_context: missing, future, or stale observation exceeds the declared maximum age"
                .into(),
        ),
    }

    let ego_state = snapshot.get("ego_state");
    let ego_frame = ego_state
        .and_then(|state| state.get("reference_frame_id"))
        .and_then(Value::as_str);
    let ego_position = ego_state
        .and_then(|state| state.get("position_m"))
        .and_then(Value::as_array);
    let ego_velocity = ego_state
        .and_then(|state| state.get("velocity_mps"))
        .and_then(Value::as_array);

    let (target_position, target_velocity, target_frame, target_scenario, target_tick, target_uncertainty) = if profile
        == "simulation_benchmark"
    {
        let actor = snapshot
            .get("simulator_truth_actor_states")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|actor| actor.get("actor_id").and_then(Value::as_str) == Some(reference_id));
        if let Some(actor) = actor {
            let actor_source = actor.get("source_ref").and_then(Value::as_str);
            if actor_source != Some(source_id) {
                errors.push(
                    "follow_target_context: simulator actor and evidence_source_ref must share the exact oracle source"
                        .into(),
                );
            }
            (
                actor.get("position_m").and_then(Value::as_array),
                actor.get("velocity_mps").and_then(Value::as_array),
                actor.get("reference_frame_id").and_then(Value::as_str),
                actor.get("scenario_id").and_then(Value::as_str),
                actor.get("acquisition_tick_ns").and_then(Value::as_u64),
                None,
            )
        } else {
            errors.push(format!(
                "follow_target_context: unknown simulator truth actor {reference_id}"
            ));
            (None, None, None, None, None, None)
        }
    } else {
        let object = snapshot
            .get("object_hypotheses")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|object| {
                object.get("hypothesis_id").and_then(Value::as_str) == Some(reference_id)
            });
        if let Some(object) = object {
            if object.get("track_lineage_id").and_then(Value::as_str)
                != context.get("actor_id").and_then(Value::as_str)
            {
                errors.push(
                    "follow_target_context: actor_id must match the linked track_lineage_id"
                        .into(),
                );
            }
            let refs: BTreeSet<&str> = object
                .get("source_refs")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            if !refs.contains(source_id) {
                errors.push(
                    "follow_target_context: evidence_source_ref must be attached to the linked object hypothesis"
                        .into(),
                );
            }
            if !matches!(
                object.get("epistemic_state").and_then(Value::as_str),
                Some("observed" | "estimated")
            ) {
                errors.push(
                    "follow_target_context: stale, unknown, occluded, predicted or contradicted object cannot be a current Follow target"
                        .into(),
                );
            }
            (
                object.get("position_m").and_then(Value::as_array),
                object.get("velocity_mps").and_then(Value::as_array),
                object.get("reference_frame_id").and_then(Value::as_str),
                snapshot_scenario,
                object
                    .get("measurement_time_tick_ns")
                    .and_then(Value::as_u64),
                object.get("uncertainty"),
            )
        } else {
            errors.push(format!(
                "follow_target_context: unknown tracked object hypothesis {reference_id}"
            ));
            (None, None, None, None, None, None)
        }
    };
    if target_frame != context_frame || target_frame != ego_frame {
        errors.push(
            "follow_target_context: ego, target and requested Follow frame must match exactly"
                .into(),
        );
    }
    if target_scenario != scenario {
        errors.push("follow_target_context: target actor must belong to the exact scenario".into());
    }
    if target_tick != observation_tick {
        errors.push(
            "follow_target_context: target state time must match the admitted observation time"
                .into(),
        );
    }

    let gap = context.get("gap_m").and_then(Value::as_f64);
    let relative_speed = context.get("relative_speed_mps").and_then(Value::as_f64);
    let uncertainty = context
        .get("distance_uncertainty_m")
        .and_then(Value::as_f64);
    if gap.is_none_or(|value| !value.is_finite() || value < 0.0)
        || relative_speed.is_none_or(|value| !value.is_finite())
        || uncertainty.is_none_or(|value| !value.is_finite() || value < 0.0)
    {
        errors.push(
            "follow_target_context: gap, relative speed and uncertainty must be finite; distance and uncertainty must be non-negative"
                .into(),
        );
    }

    if matches!(profile, "local_perception" | "deployment") {
        let ego_sigma = longitudinal_sigma_upper_bound(
            ego_state.and_then(|state| state.get("uncertainty")),
        );
        let target_sigma = longitudinal_sigma_upper_bound(target_uncertainty);
        match (ego_sigma, target_sigma, uncertainty) {
            (Some(ego_sigma), Some(target_sigma), Some(reported_bound)) => {
                let conservative_bound = ego_sigma + target_sigma;
                if reported_bound + 1e-6 < conservative_bound {
                    errors.push(format!(
                        "follow_target_context: distance_uncertainty_m understates the conservative longitudinal sigma bound ({reported_bound} < {conservative_bound})"
                    ));
                }
            }
            _ => errors.push(
                "follow_target_context: local Follow requires diagonal covariance with an x_m component for ego and target, plus a finite distance uncertainty bound"
                    .into(),
            ),
        }
    }

    if let (Some(ego_position), Some(target_position), Some(gap)) =
        (ego_position, target_position, gap)
    {
        if ego_position.len() == 3 && target_position.len() == 3 {
            if let (Some(ego_x), Some(target_x)) =
                (ego_position[0].as_f64(), target_position[0].as_f64())
            {
                let expected_gap = target_x - ego_x;
                if (gap - expected_gap).abs() > 1e-6 || expected_gap <= 0.0 {
                    errors.push(
                        "follow_target_context: gap_m must equal positive target.x - ego.x in the declared local_road frame"
                            .into(),
                    );
                }
            }
        }
    }
    if let (Some(ego_velocity), Some(target_velocity), Some(relative_speed)) =
        (ego_velocity, target_velocity, relative_speed)
    {
        if ego_velocity.len() == 3 && target_velocity.len() == 3 {
            if let (Some(ego_vx), Some(target_vx)) =
                (ego_velocity[0].as_f64(), target_velocity[0].as_f64())
            {
                let expected_relative_speed = target_vx - ego_vx;
                if (relative_speed - expected_relative_speed).abs() > 1e-6 {
                    errors.push(
                        "follow_target_context: relative_speed_mps must equal target.vx - ego.vx"
                            .into(),
                    );
                }
            }
        }
    }
}
fn validate_source_role(errors: &mut Vec<String>, source: &Value) {
    let id = source
        .get("source_id")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let role = source
        .get("source_role")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let modality = source
        .get("modality")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let evidence = source
        .get("evidence_kind")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let sensor_modality = matches!(
        modality,
        "rgb_camera"
            | "stereo_camera"
            | "lidar"
            | "radar"
            | "ultrasonic"
            | "imu"
            | "gnss"
            | "wheel_odometry"
    );
    let compatible = match role {
        "local_sensor_observation" => sensor_modality && evidence == "measurement",
        "synthetic_sensor_observation" => {
            (sensor_modality || modality == "synthetic") && evidence == "synthetic_observation"
        }
        "remote_cooperative_claim" => modality == "v2x" && evidence == "advisory",
        "simulator_oracle" => modality == "synthetic" && evidence == "synthetic_observation",
        "map_prior" => modality == "map" && evidence == "map_prior",
        "derived_estimate" => evidence == "estimate",
        "unknown" => true,
        _ => false,
    };
    if !compatible {
        errors.push(format!(
            "source {id}: source_role {role} is incompatible with modality {modality} and evidence_kind {evidence}"
        ));
    }
}

fn check_source_role_ref(
    errors: &mut Vec<String>,
    label: &str,
    reference: Option<&Value>,
    roles: &BTreeMap<String, String>,
    allowed: &[&str],
) {
    let Some(id) = reference.and_then(Value::as_str) else {
        return;
    };
    if let Some(role) = roles.get(id)
        && !allowed.contains(&role.as_str())
    {
        errors.push(format!(
            "{label}: source {id} has role {role}, which is not allowed here"
        ));
    }
}

fn check_source_role_refs(
    errors: &mut Vec<String>,
    label: &str,
    references: Option<&Value>,
    roles: &BTreeMap<String, String>,
    allowed: &[&str],
) {
    if let Some(references) = references.and_then(Value::as_array) {
        for reference in references {
            check_source_role_ref(errors, label, Some(reference), roles, allowed);
        }
    }
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

    if order_len == 0
        || order_len != units_len
        || diagonal.is_none_or(|values| values.len() != order_len)
    {
        errors.push(format!(
            "{prefix}.uncertainty: diagonal covariance dimension must match state_order and units"
        ));
        return;
    }

    if let Some(values) = diagonal
        && values.iter().any(|value| {
            value
                .as_f64()
                .is_none_or(|number| !number.is_finite() || number < 0.0)
        })
    {
        errors.push(format!(
            "{prefix}.uncertainty: diagonal covariance entries must be finite and non-negative"
        ));
    }
}

fn validate_class_distribution(errors: &mut Vec<String>, prefix: &str, classes: Option<&Value>) {
    let Some(classes) = classes else {
        return;
    };
    if classes.get("distribution_status").and_then(Value::as_str) != Some("calibrated_probability")
    {
        return;
    }
    let scores = classes.get("scores").and_then(Value::as_array);
    let Some(scores) = scores else {
        errors.push(format!(
            "{prefix}.class_hypotheses: calibrated probability requires scores"
        ));
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
    fn rejects_remote_cooperative_claim_as_local_track_source() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["object_hypotheses"][0]["source_refs"][0] =
            Value::String("cooperative-v2x-claim-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_simulator_oracle_as_local_track_source() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["object_hypotheses"][0]["source_refs"][0] =
            Value::String("simulator-oracle-truth-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_local_sensor_in_cooperative_advisory_refs() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["cooperative_advisory_refs"][0] =
            Value::String("front-lidar-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_local_sensor_in_simulator_truth_refs() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["simulator_truth_refs"][0] = Value::String("front-camera-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_v2x_modality_mislabeled_as_local_sensor() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["sources"][2]["source_role"] =
            Value::String("local_sensor_observation".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn accepts_simulator_oracle_follow_target_only_in_simulation_benchmark() {
        let (schema, fixture) = load_fixture();
        validate_values(&schema, &fixture)
            .expect("simulator oracle should be admissible for a simulation benchmark");
    }

    #[test]
    fn rejects_simulator_oracle_follow_target_for_deployment() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["consumer_profile"] =
            Value::String("deployment".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_remote_cooperative_claim_as_follow_target() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("cooperative-v2x-claim-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn accepts_synthetic_local_perception_follow_target_linked_to_track() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        context["distance_uncertainty_m"] = Value::from(0.7);
        validate_values(&schema, &fixture)
            .expect("synthetic sensor observations may test the local perception adapter path");
    }

    #[test]
    fn rejects_synthetic_local_perception_evidence_for_deployment() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("deployment".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_stale_local_sensor_follow_target() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        fixture["snapshot"]["sources"][4]["currentness"] = Value::String("stale".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_follow_target_from_wrong_coordinate_frame() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["coordinate_frame_id"] =
            Value::String("front-camera-image-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_follow_target_with_unlinked_object_hypothesis() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("missing-hypothesis".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_simulator_truth_actor_with_unregistered_source() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["simulator_truth_actor_states"][0]["source_ref"] =
            Value::String("front-lidar-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_simulator_truth_actor_from_another_scenario() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["simulator_truth_actor_states"][0]["scenario_id"] =
            Value::String("different-scenario".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_follow_target_link_to_unknown_simulator_actor() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["target_reference_id"] =
            Value::String("oracle-actor-missing".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_follow_target_gap_scalar_not_supported_by_scene_geometry() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["gap_m"] = Value::from(99.0);
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_follow_target_relative_speed_not_supported_by_scene_geometry() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["relative_speed_mps"] = Value::from(25.0);
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_local_follow_uncertainty_below_uncorrelated_covariance_bound() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        context["distance_uncertainty_m"] = Value::from(0.69);
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn accepts_local_follow_uncertainty_at_conservative_covariance_bound() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        context["distance_uncertainty_m"] = Value::from(0.7);
        validate_values(&schema, &fixture)
            .expect("reported gap uncertainty should meet the conservative covariance bound");
    }

    #[test]
    fn rejects_follow_target_actor_id_that_does_not_match_linked_track() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("wrong-track".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_non_actor_sensor_modality_as_local_follow_evidence() {
        let (schema, mut fixture) = load_fixture();
        let context = &mut fixture["snapshot"]["follow_target_context"];
        context["consumer_profile"] = Value::String("local_perception".into());
        context["evidence_source_ref"] = Value::String("local-lidar-track-fixture-01".into());
        context["actor_id"] = Value::String("track-42".into());
        context["target_reference_kind"] = Value::String("tracked_object_hypothesis".into());
        context["target_reference_id"] = Value::String("object-hyp-42".into());
        fixture["snapshot"]["sources"][4]["modality"] = Value::String("imu".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_follow_target_past_declared_maximum_age() {
        let (schema, mut fixture) = load_fixture();
        let stale_tick = Value::from(11_000_000_000u64);
        fixture["snapshot"]["follow_target_context"]["max_age_ns"] = Value::from(1u64);
        fixture["snapshot"]["follow_target_context"]["observation_tick_ns"] = stale_tick.clone();
        fixture["snapshot"]["sources"][3]["acquisition_tick_ns"] = stale_tick.clone();
        fixture["snapshot"]["simulator_truth_actor_states"][0]["acquisition_tick_ns"] = stale_tick;
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn unknown_follow_target_cannot_retain_oracle_or_remote_fallback() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["status"] = Value::String("unknown".into());
        for key in [
            "evidence_source_ref",
            "scenario_id",
            "actor_id",
            "coordinate_frame_id",
            "observation_tick_ns",
            "max_age_ns",
            "gap_m",
            "relative_speed_mps",
            "distance_uncertainty_m",
            "target_reference_kind",
            "target_reference_id",
        ] {
            fixture["snapshot"]["follow_target_context"][key] = Value::Null;
        }
        validate_values(&schema, &fixture)
            .expect("unknown target must be explicit and contain no fallback evidence");

        fixture["snapshot"]["follow_target_context"]["evidence_source_ref"] =
            Value::String("simulator-oracle-truth-01".into());
        assert!(validate_values(&schema, &fixture).is_err());
    }

    #[test]
    fn rejects_future_dated_follow_target_observation() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["follow_target_context"]["observation_tick_ns"] =
            Value::from(13_000_000_000u64);
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
    fn allows_multiple_hypotheses_for_one_track_lineage() {
        let (schema, mut fixture) = load_fixture();
        fixture["snapshot"]["object_hypotheses"][1]["track_lineage_id"] =
            Value::String("track-42".into());
        assert!(validate_values(&schema, &fixture).is_ok());
    }

    #[test]
    fn rejects_duplicate_hypothesis_identity() {
        let (schema, mut fixture) = load_fixture();
        let duplicate_id = fixture["snapshot"]["object_hypotheses"][0]["hypothesis_id"].clone();
        fixture["snapshot"]["object_hypotheses"][1]["hypothesis_id"] = duplicate_id;
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

    fn load_traffic_fixtures() -> (Value, Value, Value) {
        let root = repo_root();
        let schema: Value = serde_json::from_str(
            &fs::read_to_string(
                root.join("docs/autonomy/SAM_TRAFFIC_CONVENTION_PROFILE_SCHEMA_V0_1.json"),
            )
            .expect("traffic schema text"),
        )
        .expect("traffic schema JSON");
        let profiles: Value =
            serde_json::from_str(
                &fs::read_to_string(root.join(
                    "docs/autonomy/fixtures/SAM_TRAFFIC_CONVENTION_PROFILE_FIXTURE_V0_1.json",
                ))
                .expect("traffic profile fixture text"),
            )
            .expect("traffic profile fixture JSON");
        let mirrored: Value = serde_json::from_str(
            &fs::read_to_string(
                root.join("docs/autonomy/fixtures/SAM_TRAFFIC_CONVENTION_METAMORPHIC_V0_1.json"),
            )
            .expect("traffic metamorphic fixture text"),
        )
        .expect("traffic metamorphic fixture JSON");
        (schema, profiles, mirrored)
    }

    #[test]
    fn traffic_profile_registry_conforms_to_schema_and_sign_semantics() {
        let (schema, profiles, _) = load_traffic_fixtures();
        validate_traffic_values(&schema, &profiles).expect("traffic profile registry must conform");
    }

    #[test]
    fn mirrored_left_right_scene_pair_is_equivariant() {
        let (schema, _, mirrored) = load_traffic_fixtures();
        validate_traffic_values(&schema, &mirrored)
            .expect("paired LHT/RHT scenes must satisfy mirror invariants");
    }

    #[test]
    fn rejects_scene_profile_that_conflicts_with_lane_geometry() {
        let (schema, _, mut mirrored) = load_traffic_fixtures();
        mirrored["paired_scenes"][0]["traffic_convention_profile_id"] =
            Value::String("synthetic-rht-lhd-v1".into());
        mirrored["paired_scenes"][0]["map_traffic_convention_profile_id"] =
            Value::String("synthetic-rht-lhd-v1".into());
        mirrored["paired_scenes"][0]["travel_frame_id"] = Value::String("road-local-rht-v1".into());
        assert!(validate_traffic_values(&schema, &mirrored).is_err());
    }

    #[test]
    fn rejects_untransformed_steering_under_road_reflection() {
        let (schema, _, mut mirrored) = load_traffic_fixtures();
        mirrored["paired_scenes"][1]["ego_state"]["steering_angle_rad"] = Value::from(0.1);
        mirrored["paired_scenes"][1]["candidate_command"]["steering"] = Value::from(0.2);
        assert!(validate_traffic_values(&schema, &mirrored).is_err());
    }

    #[test]
    fn steering_wheel_position_does_not_change_traffic_side_semantics() {
        let (schema, _, mut mirrored) = load_traffic_fixtures();
        mirrored["profiles"][0]["steering_wheel_position"] = Value::String("left".into());
        assert!(validate_traffic_values(&schema, &mirrored).is_ok());
    }

    #[test]
    fn rejects_scene_map_traffic_profile_mismatch() {
        let (schema, _, mut mirrored) = load_traffic_fixtures();
        mirrored["paired_scenes"][0]["map_traffic_convention_profile_id"] =
            Value::String("synthetic-rht-lhd-v1".into());
        assert!(validate_traffic_values(&schema, &mirrored).is_err());
    }

    #[test]
    fn rejects_passing_side_without_matching_rule_profile() {
        let (schema, _, mut mirrored) = load_traffic_fixtures();
        mirrored["paired_scenes"][0]["declared_passing_side_sign"] = Value::from(1);
        assert!(validate_traffic_values(&schema, &mirrored).is_err());
    }

    #[test]
    fn unknown_traffic_side_cannot_retain_passing_authority() {
        let (schema, mut profiles, _) = load_traffic_fixtures();
        profiles["profiles"][0]["traffic_side"] = Value::String("unknown".into());
        profiles["profiles"][0]["passing_side_sign"] = Value::Null;
        profiles["profiles"][0]["passing_rule_profile_id"] = Value::Null;
        profiles["profiles"][0]["passing_rule_profile_version"] = Value::Null;
        profiles["profiles"][0]["applicability"] = Value::String("unknown".into());
        assert!(validate_traffic_values(&schema, &profiles).is_ok());

        profiles["profiles"][0]["passing_side_sign"] = Value::from(-1);
        assert!(validate_traffic_values(&schema, &profiles).is_err());
    }

    #[test]
    fn rejects_verified_traffic_profile_without_evidence_records() {
        let (schema, mut profiles, _) = load_traffic_fixtures();
        profiles["profiles"][0]["applicability"] =
            Value::String("verified_for_declared_context".into());
        profiles["profiles"][0]["verification_evidence_refs"] =
            Value::Array(vec![Value::String("law-evidence-1".into())]);
        profiles["profiles"][0]["verification_method_id"] =
            Value::String("independent-rule-review-v1".into());
        profiles["profiles"][0]["verification_record_id"] =
            Value::String("review-2026-11-08-001".into());
        profiles["profiles"][0]["verification_valid_from"] = Value::String("2026-01-01".into());
        profiles["profiles"][0]["verification_valid_until"] = Value::String("2026-12-31".into());
        assert!(validate_traffic_values(&schema, &profiles).is_err());
    }

    #[test]
    fn accepts_verified_traffic_profile_with_matching_rule_evidence() {
        let (schema, mut profiles, _) = load_traffic_fixtures();
        profiles["profiles"][0]["applicability"] =
            Value::String("verified_for_declared_context".into());
        profiles["profiles"][0]["verification_evidence_refs"] =
            Value::Array(vec![Value::String("law-evidence-1".into())]);
        profiles["profiles"][0]["verification_method_id"] =
            Value::String("independent-rule-review-v1".into());
        profiles["profiles"][0]["verification_record_id"] =
            Value::String("review-2026-11-08-001".into());
        profiles["profiles"][0]["verification_valid_from"] = Value::String("2026-01-01".into());
        profiles["profiles"][0]["verification_valid_until"] = Value::String("2026-12-31".into());
        profiles["evidence_records"] = Value::Array(vec![serde_json::json!({
            "evidence_id":"law-evidence-1",
            "jurisdiction_profile_id":"synthetic-left-traffic-demo",
            "jurisdiction_profile_version":"synthetic-1.0.0",
            "evidence_kind":"traffic_law",
            "issuer":"Synthetic Authority",
            "title":"Synthetic traffic-side rule profile",
            "reference_uri":"synthetic-evidence://lht/profile-v1",
            "content_digest":format!("sha256:{}", "a".repeat(64)),
            "published_at":"2025-12-01",
            "effective_from":"2025-12-01",
            "effective_until":"2027-12-31",
            "retrieved_at":"2026-11-08"
        })]);
        validate_traffic_values(&schema, &profiles)
            .expect("verified profile with matching rule evidence should validate");
    }

    #[test]
    fn rejects_verified_traffic_profile_evidence_for_another_jurisdiction() {
        let (schema, mut profiles, _) = load_traffic_fixtures();
        profiles["profiles"][0]["applicability"] =
            Value::String("verified_for_declared_context".into());
        profiles["profiles"][0]["verification_evidence_refs"] =
            Value::Array(vec![Value::String("law-evidence-1".into())]);
        profiles["profiles"][0]["verification_method_id"] =
            Value::String("independent-rule-review-v1".into());
        profiles["profiles"][0]["verification_record_id"] =
            Value::String("review-2026-11-08-001".into());
        profiles["profiles"][0]["verification_valid_from"] = Value::String("2026-01-01".into());
        profiles["profiles"][0]["verification_valid_until"] = Value::String("2026-12-31".into());
        profiles["evidence_records"] = Value::Array(vec![serde_json::json!({
            "evidence_id":"law-evidence-1",
            "jurisdiction_profile_id":"different-jurisdiction",
            "jurisdiction_profile_version":"synthetic-1.0.0",
            "evidence_kind":"traffic_law",
            "issuer":"Synthetic Authority",
            "title":"Wrong jurisdiction source",
            "reference_uri":"synthetic-evidence://wrong-jurisdiction",
            "content_digest":format!("sha256:{}", "b".repeat(64)),
            "published_at":"2025-12-01",
            "effective_from":"2025-12-01",
            "effective_until":"2027-12-31",
            "retrieved_at":"2026-11-08"
        })]);
        assert!(validate_traffic_values(&schema, &profiles).is_err());
    }

    #[test]
    fn scene_traffic_profile_reference_resolves_across_contracts() {
        let (scene_schema, scene_fixture) = load_fixture();
        let (traffic_schema, _, traffic_fixture) = load_traffic_fixtures();
        validate_bundle_values(
            &scene_schema,
            &scene_fixture,
            &traffic_schema,
            &traffic_fixture,
        )
        .expect("scene traffic-convention profile must resolve in the profile registry");
    }

    #[test]
    fn rejects_scene_traffic_profile_missing_from_registry() {
        let (scene_schema, mut scene_fixture) = load_fixture();
        let (traffic_schema, _, traffic_fixture) = load_traffic_fixtures();
        scene_fixture["snapshot"]["traffic_convention_profile_id"] =
            Value::String("missing-traffic-profile".into());
        assert!(
            validate_bundle_values(
                &scene_schema,
                &scene_fixture,
                &traffic_schema,
                &traffic_fixture,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_scene_traffic_profile_frame_missing_from_scene_registry() {
        let (scene_schema, scene_fixture) = load_fixture();
        let (traffic_schema, _, mut traffic_fixture) = load_traffic_fixtures();
        traffic_fixture["profiles"][0]["travel_frame_id"] =
            Value::String("unregistered-road-frame".into());
        assert!(
            validate_bundle_values(
                &scene_schema,
                &scene_fixture,
                &traffic_schema,
                &traffic_fixture,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_traffic_profile_bound_to_non_road_frame() {
        let (scene_schema, scene_fixture) = load_fixture();
        let (traffic_schema, _, mut traffic_fixture) = load_traffic_fixtures();
        traffic_fixture["profiles"][0]["travel_frame_id"] =
            Value::String("front-camera-image-01".into());
        assert!(
            validate_bundle_values(
                &scene_schema,
                &scene_fixture,
                &traffic_schema,
                &traffic_fixture,
            )
            .is_err()
        );
    }
}
