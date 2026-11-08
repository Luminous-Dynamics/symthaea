use std::collections::BTreeSet;
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
                frames.iter().find(|frame| {
                    frame.get("frame_id").and_then(Value::as_str) == profile_frame_id
                })
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
        if traffic_side == Some("unknown")
            && (profile.get("passing_side_sign") != Some(&Value::Null)
                || profile.get("applicability").and_then(Value::as_str) != Some("unknown"))
        {
            errors.push(format!(
                "profiles[{index}]: unknown traffic side must have no passing-side authority and unknown applicability"
            ));
        }
        if profile
            .get("passing_rule_profile_id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
            || profile
                .get("passing_rule_profile_version")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            errors.push(format!(
                "profiles[{index}]: passing-side semantics require an explicit rule-profile ID and version"
            ));
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
        profiles["profiles"][0]["applicability"] = Value::String("unknown".into());
        assert!(validate_traffic_values(&schema, &profiles).is_ok());

        profiles["profiles"][0]["passing_side_sign"] = Value::from(-1);
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
