use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashSet},
    env, fs,
    process::ExitCode,
};

const EXPECTED_FIXTURES: usize = 24;
const MINIMUM_SAMPLE_COUNT: usize = 20;
const ECE_BINS: usize = 10;
const LOG_EPSILON: f64 = 1e-15;

#[derive(Debug, Clone, PartialEq)]
struct ReliabilityBin {
    lower: f64,
    upper: f64,
    count: usize,
    mean_probability: f64,
    observed_frequency: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct CalibrationMetrics {
    sample_count: usize,
    independent_group_count: usize,
    uncertainty_interval_status: &'static str,
    brier_score: f64,
    log_loss: f64,
    ece_equal_width: f64,
    aurc: f64,
    reliability_bins: Vec<ReliabilityBin>,
    risk_at_coverage: Vec<(f64, f64)>,
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|value| !value.trim().is_empty())
}

fn string_set(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn add(errors: &mut Vec<String>, code: &str) {
    errors.push(code.to_string());
}

fn set_at_path(root: &mut Value, path: &str, replacement: Value) -> bool {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return false;
    }
    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        current = match current {
            Value::Object(map) => match map.get_mut(*part) {
                Some(value) => value,
                None => return false,
            },
            Value::Array(items) => match part.parse::<usize>().ok().and_then(|i| items.get_mut(i)) {
                Some(value) => value,
                None => return false,
            },
            _ => return false,
        };
    }
    let last = parts[parts.len() - 1];
    match current {
        Value::Object(map) => {
            if !map.contains_key(last) {
                return false;
            }
            map.insert(last.to_string(), replacement);
            true
        }
        Value::Array(items) => match last.parse::<usize>() {
            Ok(index) if index < items.len() => {
                items[index] = replacement;
                true
            }
            _ => false,
        },
        _ => false,
    }
}

fn remove_at_path(root: &mut Value, path: &str) -> bool {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return false;
    }
    let mut current = root;
    for part in &parts[..parts.len() - 1] {
        current = match current {
            Value::Object(map) => match map.get_mut(*part) {
                Some(value) => value,
                None => return false,
            },
            Value::Array(items) => match part.parse::<usize>().ok().and_then(|i| items.get_mut(i)) {
                Some(value) => value,
                None => return false,
            },
            _ => return false,
        };
    }
    let last = parts[parts.len() - 1];
    match current {
        Value::Object(map) => map.remove(last).is_some(),
        Value::Array(items) => match last.parse::<usize>() {
            Ok(index) if index < items.len() => {
                items.remove(index);
                true
            }
            _ => false,
        },
        _ => false,
    }
}

fn set_all_at_path(root: &mut Value, path: &str, replacement: Value) -> bool {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if parts.len() != 3 || parts[1] != "*" {
        return false;
    }
    let Some(rows) = root.get_mut(parts[0]).and_then(Value::as_array_mut) else {
        return false;
    };
    for row in rows {
        let Some(object) = row.as_object_mut() else {
            return false;
        };
        if !object.contains_key(parts[2]) {
            return false;
        }
        object.insert(parts[2].to_string(), replacement.clone());
    }
    true
}

fn digest_json(value: &Value) -> Option<String> {
    let bytes = serde_json::to_vec(value).ok()?;
    let hash = Sha256::digest(bytes);
    Some(format!("{hash:x}"))
}


const SUPPORTED_SCHEMA_KEYWORDS: [&str; 16] = [
    "$schema", "$id", "title", "type", "additionalProperties", "required",
    "properties", "const", "enum", "minLength", "pattern", "minimum",
    "minItems", "uniqueItems", "items", "maximum",
];

fn validate_schema_definition(schema: &Value, context: &str, errors: &mut Vec<String>) {
    let Some(object) = schema.as_object() else {
        add(errors, &format!("{context}:schema_node_not_object"));
        return;
    };
    for keyword in object.keys() {
        if !SUPPORTED_SCHEMA_KEYWORDS.contains(&keyword.as_str()) {
            add(errors, &format!("{context}:unsupported_schema_keyword:{keyword}"));
        }
    }
    if let Some(kind) = schema.get("type") {
        let supported = ["object", "array", "string", "number", "integer", "boolean", "null"];
        let types: Vec<&str> = if let Some(single) = kind.as_str() {
            vec![single]
        } else {
            kind.as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect()
        };
        if types.is_empty() || types.iter().any(|item| !supported.contains(item)) {
            add(errors, &format!("{context}:unsupported_schema_type"));
        }
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        if pattern != "^[a-f0-9]{64}$" {
            add(errors, &format!("{context}:unsupported_schema_pattern"));
        }
    }
    if let Some(properties) = schema.get("properties") {
        if let Some(properties) = properties.as_object() {
            for (name, child) in properties {
                validate_schema_definition(child, &format!("{context}.{name}"), errors);
            }
        } else {
            add(errors, &format!("{context}:schema_properties_not_object"));
        }
    }
    if let Some(items) = schema.get("items") {
        validate_schema_definition(items, &format!("{context}[]"), errors);
    }
}

fn schema_type_matches(value: &Value, kind: &str) -> bool {
    match kind {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn validate_json_schema(value: &Value, schema: &Value, context: &str, errors: &mut Vec<String>) {
    if let Some(kind) = schema.get("type") {
        let matches = if let Some(single) = kind.as_str() {
            schema_type_matches(value, single)
        } else {
            kind.as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .any(|candidate| schema_type_matches(value, candidate))
        };
        if !matches {
            add(errors, &format!("{context}:schema_type_mismatch"));
            return;
        }
    }
    if let Some(expected) = schema.get("const") {
        if value != expected {
            add(errors, &format!("{context}:schema_const_mismatch"));
        }
    }
    if let Some(variants) = schema.get("enum").and_then(Value::as_array) {
        if !variants.contains(value) {
            add(errors, &format!("{context}:schema_enum_mismatch"));
        }
    }
    if let Some(text) = value.as_str() {
        if let Some(minimum) = schema.get("minLength").and_then(Value::as_u64) {
            if text.chars().count() < minimum as usize {
                add(errors, &format!("{context}:schema_min_length"));
            }
        }
        if schema.get("pattern").is_some() && !is_sha256(text) {
            add(errors, &format!("{context}:schema_pattern_mismatch"));
        }
    }
    if let Some(number) = value.as_f64() {
        if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64) {
            if number < minimum {
                add(errors, &format!("{context}:schema_minimum"));
            }
        }
        if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64) {
            if number > maximum {
                add(errors, &format!("{context}:schema_maximum"));
            }
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(minimum) = schema.get("minItems").and_then(Value::as_u64) {
            if items.len() < minimum as usize {
                add(errors, &format!("{context}:schema_min_items"));
            }
        }
        if schema.get("uniqueItems").and_then(Value::as_bool) == Some(true) {
            let serialized: BTreeSet<String> =
                items.iter().map(Value::to_string).collect();
            if serialized.len() != items.len() {
                add(errors, &format!("{context}:schema_unique_items"));
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in items.iter().enumerate() {
                validate_json_schema(item, item_schema, &format!("{context}[{index}]"), errors);
            }
        }
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for property in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(property) {
                    add(errors, &format!("{context}:schema_required_missing:{property}"));
                }
            }
        }
        let properties = schema.get("properties").and_then(Value::as_object);
        if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
            for property in object.keys() {
                if !properties.is_some_and(|known| known.contains_key(property)) {
                    add(errors, &format!("{context}:schema_additional_property:{property}"));
                }
            }
        }
        if let Some(properties) = properties {
            for (name, child_schema) in properties {
                if let Some(child_value) = object.get(name) {
                    validate_json_schema(
                        child_value,
                        child_schema,
                        &format!("{context}.{name}"),
                        errors,
                    );
                }
            }
        }
    }
}

fn validate_schema_header(
    schema: &Value,
    expected_id: &str,
    context: &str,
    errors: &mut Vec<String>,
) {
    if schema["$schema"].as_str()
        != Some("https://json-schema.org/draft/2020-12/schema")
    {
        add(errors, &format!("{context}:schema_dialect_invalid"));
    }
    if schema["$id"].as_str() != Some(expected_id) {
        add(errors, &format!("{context}:schema_id_mismatch"));
    }
    if schema["type"].as_str() != Some("object")
        || schema["additionalProperties"].as_bool() != Some(false)
    {
        add(errors, &format!("{context}:schema_root_contract_invalid"));
    }
}

fn validate_schema_pair(
    evaluation: &Value,
    evaluation_schema: &Value,
    controls: &Value,
    controls_schema: &Value,
) -> Vec<String> {
    let mut errors = Vec::new();
    validate_schema_definition(evaluation_schema, "evaluation_schema", &mut errors);
    validate_schema_definition(controls_schema, "controls_schema", &mut errors);
    validate_schema_header(
        evaluation_schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-confidence-calibration-001/CONFIDENCE_EVALUATION_V0_1.schema.json",
        "evaluation_schema",
        &mut errors,
    );
    validate_schema_header(
        controls_schema,
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/research/ig-confidence-calibration-001/CONFIDENCE_NEGATIVE_CONTROLS_V0_1.schema.json",
        "controls_schema",
        &mut errors,
    );
    validate_json_schema(evaluation, evaluation_schema, "evaluation", &mut errors);
    validate_json_schema(controls, controls_schema, "negative_controls", &mut errors);
    errors.sort();
    errors.dedup();
    errors
}

fn validate_bundle(bundle: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    if field(bundle, "schema_version") != Some("ig-confidence-evaluation-v0.1")
        || field(bundle, "benchmark_id") != Some("IG-CONFIDENCE-CALIBRATION-001")
    {
        add(&mut errors, "evaluation_identity_invalid");
    }
    if field(bundle, "status") != Some("synthetic_known_answer_unexecuted") {
        add(&mut errors, "evaluation_status_invalid");
    }

    let profile = &bundle["calibration_profile"];
    if field(profile, "status") != Some("Calibrated") {
        add(&mut errors, "profile_not_calibrated");
    }
    if !is_sha256(field(profile, "calibration_profile_sha256").unwrap_or_default()) {
        add(&mut errors, "profile_digest_invalid");
    }
    if !is_sha256(field(profile, "calibration_report_sha256").unwrap_or_default()) {
        add(&mut errors, "report_digest_invalid");
    }
    if !matches!(
        field(profile, "method"),
        Some("PlattScaling" | "IsotonicRegression" | "BetaCalibration" | "TemperatureScaling")
    ) {
        add(&mut errors, "calibration_method_unsupported");
    }
    if field(profile, "scale_id") != Some("probability_of_correctness_v1") {
        add(&mut errors, "confidence_scale_unsupported");
    }

    let evaluation = &bundle["evaluation_identity"];
    let model = &profile["model_identity"];
    if field(model, "provider") != field(evaluation, "provider")
        || field(model, "model") != field(evaluation, "model")
        || field(model, "revision") != field(evaluation, "revision")
        || field(model, "provider").is_none()
    {
        add(&mut errors, "model_identity_mismatch");
    }
    if field(profile, "prompt_family_sha256")
        != field(evaluation, "prompt_family_sha256")
        || !is_sha256(field(profile, "prompt_family_sha256").unwrap_or_default())
    {
        add(&mut errors, "prompt_family_mismatch");
    }
    if field(profile, "task_scope") != field(evaluation, "task_scope")
        || field(evaluation, "task_scope") != Some("factual_qa_binary_correctness")
    {
        add(&mut errors, "task_scope_mismatch");
    }
    if field(evaluation, "task_truth_profile") != Some("SyntheticKnownAnswer") {
        add(&mut errors, "evaluation_truth_profile_unsupported");
    }

    let calibration_data = &profile["calibration_data"];
    let fit_digest = field(calibration_data, "fit_dataset_sha256").unwrap_or_default();
    let validation_digest =
        field(calibration_data, "validation_dataset_sha256").unwrap_or_default();
    let evaluation_digest = field(bundle, "evaluation_dataset_sha256").unwrap_or_default();
    if !is_sha256(fit_digest) || !is_sha256(validation_digest) || !is_sha256(evaluation_digest) {
        add(&mut errors, "calibration_dataset_digest_invalid");
    } else if fit_digest == validation_digest
        || fit_digest == evaluation_digest
        || validation_digest == evaluation_digest
    {
        add(&mut errors, "calibration_dataset_digest_reuse");
    }

    let fit_groups: BTreeSet<String> = calibration_data["fit_group_ids"]
        .as_array().into_iter().flatten().filter_map(Value::as_str)
        .map(ToOwned::to_owned).collect();
    let validation_groups: BTreeSet<String> = calibration_data["validation_group_ids"]
        .as_array().into_iter().flatten().filter_map(Value::as_str)
        .map(ToOwned::to_owned).collect();
    let evaluation_groups: BTreeSet<String> = calibration_data["evaluation_group_ids"]
        .as_array().into_iter().flatten().filter_map(Value::as_str)
        .map(ToOwned::to_owned).collect();
    if !fit_groups.is_disjoint(&validation_groups) {
        add(&mut errors, "fit_validation_group_overlap");
    }
    if !fit_groups.is_disjoint(&evaluation_groups) {
        add(&mut errors, "fit_evaluation_group_overlap");
    }
    if !validation_groups.is_disjoint(&evaluation_groups) {
        add(&mut errors, "validation_evaluation_group_overlap");
    }

    if calibration_data["sample_count"].as_u64().unwrap_or(0) < MINIMUM_SAMPLE_COUNT as u64 {
        add(&mut errors, "calibration_sample_count_insufficient");
    }
    if field(&profile["validator"], "identity").is_none() {
        add(&mut errors, "validator_identity_missing");
    }
    if field(&profile["validator"], "version").is_none() {
        add(&mut errors, "validator_version_missing");
    }

    let Some(observations) = bundle["observations"].as_array() else {
        add(&mut errors, "evaluation_observations_missing");
        errors.sort();
        errors.dedup();
        return errors;
    };
    let minimum = bundle["minimum_observations"].as_u64().unwrap_or(MINIMUM_SAMPLE_COUNT as u64);
    if observations.len() < minimum as usize || observations.len() < MINIMUM_SAMPLE_COUNT {
        add(&mut errors, "evaluation_observation_count_insufficient");
    }
    let mut sample_ids = HashSet::new();
    let mut seen_true = false;
    let mut seen_false = false;
    for row in observations {
        let sample_id = field(row, "sample_id");
        if sample_id.is_none() || !sample_ids.insert(sample_id.unwrap().to_string()) {
            add(&mut errors, "duplicate_sample_id");
        }
        let probability = row["probability_correct"].as_f64();
        if probability.is_none_or(|p| !p.is_finite() || !(0.0..=1.0).contains(&p)) {
            add(&mut errors, "probability_out_of_range");
        }
        match row["outcome_correct"].as_bool() {
            Some(true) => seen_true = true,
            Some(false) => seen_false = true,
            None => add(&mut errors, "binary_outcome_missing"),
        }
        if field(row, "task_group_id").is_none()
            || !evaluation_groups.contains(field(row, "task_group_id").unwrap_or_default())
        {
            add(&mut errors, "evaluation_group_unbound");
        }
    }
    if !seen_true || !seen_false {
        add(&mut errors, "single_outcome_class");
    }

    if let Some(actual_digest) = digest_json(&bundle["observations"]) {
        if actual_digest != evaluation_digest {
            add(&mut errors, "evaluation_dataset_digest_mismatch");
        }
    } else {
        add(&mut errors, "evaluation_dataset_digest_unavailable");
    }

    errors.sort();
    errors.dedup();
    errors
}

fn calculate_metrics(bundle: &Value) -> Option<CalibrationMetrics> {
    let rows = bundle["observations"].as_array()?;
    if rows.is_empty() {
        return None;
    }
    let mut pairs = Vec::with_capacity(rows.len());
    let mut brier = 0.0;
    let mut log_loss = 0.0;
    for row in rows {
        let p = row["probability_correct"].as_f64()?;
        let y = row["outcome_correct"].as_bool()? as u8 as f64;
        let error = p - y;
        brier += error * error;
        let clipped = p.clamp(LOG_EPSILON, 1.0 - LOG_EPSILON);
        log_loss -= y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln();
        pairs.push((p, y));
    }
    let n = pairs.len() as f64;
    brier /= n;
    log_loss /= n;

    let mut bins = Vec::new();
    let mut ece = 0.0;
    for index in 0..ECE_BINS {
        let lower = index as f64 / ECE_BINS as f64;
        let upper = (index + 1) as f64 / ECE_BINS as f64;
        let members: Vec<(f64, f64)> = pairs
            .iter()
            .copied()
            .filter(|(probability, _)| {
                let bucket = ((*probability * ECE_BINS as f64).floor() as usize)
                    .min(ECE_BINS - 1);
                bucket == index
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        let count = members.len();
        let mean_probability =
            members.iter().map(|(probability, _)| *probability).sum::<f64>() / count as f64;
        let observed_frequency =
            members.iter().map(|(_, outcome)| *outcome).sum::<f64>() / count as f64;
        ece += (count as f64 / n) * (mean_probability - observed_frequency).abs();
        bins.push(ReliabilityBin {
            lower,
            upper,
            count,
            mean_probability,
            observed_frequency,
        });
    }

    // Equal-confidence observations are a single ranking block. Fractional
    // inclusion within a tie uses that block's empirical error rate, so row
    // order cannot alter risk@coverage or AURC.
    let mut ranked = pairs.clone();
    ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut tie_groups: Vec<(f64, usize, f64)> = Vec::new();
    for (probability, outcome) in ranked {
        if let Some((last_probability, count, error_rate)) = tie_groups.last_mut() {
            if *last_probability == probability {
                let errors_before = *error_rate * *count as f64;
                *count += 1;
                *error_rate = (errors_before + (1.0 - outcome)) / *count as f64;
                continue;
            }
        }
        tie_groups.push((probability, 1, 1.0 - outcome));
    }

    let mut observations_before = 0usize;
    let mut errors_before = 0.0;
    let mut cumulative_risk_sum = 0.0;
    for (_, group_count, group_error_rate) in &tie_groups {
        for included_from_tie in 1..=*group_count {
            let expected_tie_errors = *group_error_rate * included_from_tie as f64;
            let included = observations_before as f64 + included_from_tie as f64;
            let risk = (errors_before + expected_tie_errors) / included;
            cumulative_risk_sum += risk;
        }
        errors_before += *group_error_rate * *group_count as f64;
        observations_before += *group_count;
    }
    let aurc = cumulative_risk_sum / n;

    let mut risk_at_coverage = Vec::new();
    for coverage in [0.25, 0.50, 0.75, 1.00] {
        let target_count = coverage * n;
        let mut included = 0.0;
        let mut expected_errors = 0.0;
        for (_, group_count, group_error_rate) in &tie_groups {
            let available = *group_count as f64;
            let take = (target_count - included).clamp(0.0, available);
            expected_errors += take * *group_error_rate;
            included += take;
            if included >= target_count {
                break;
            }
        }
        if included <= 0.0 {
            return None;
        }
        risk_at_coverage.push((coverage, expected_errors / included));
    }

    let independent_group_count = bundle["observations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["task_group_id"].as_str())
        .collect::<BTreeSet<_>>()
        .len();

    Some(CalibrationMetrics {
        sample_count: rows.len(),
        independent_group_count,
        // The two-group synthetic fixture is not enough to justify a cluster
        // uncertainty interval. Confirmatory studies must preregister this.
        uncertainty_interval_status: "NotEstimated",
        brier_score: brier,
        log_loss,
        ece_equal_width: ece,
        aurc,
        reliability_bins: bins,
        risk_at_coverage,
    })
}

fn read_json(path: &str) -> Result<Value, String> {
    let contents = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    serde_json::from_str(&contents).map_err(|error| format!("{path}: invalid JSON: {error}"))
}

fn apply_fixture(bundle: &mut Value, fixture: &Value) -> bool {
    let Some(mutation) = fixture.get("mutation") else { return true; };
    if mutation.is_null() { return true; }
    let path = field(mutation, "path").unwrap_or_default();
    match field(mutation, "operation") {
        Some("set") => set_at_path(bundle, path, mutation["value"].clone()),
        Some("remove") => remove_at_path(bundle, path),
        Some("set_all") => set_all_at_path(bundle, path, mutation["value"].clone()),
        _ => false,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: ig-confidence-calibration-qualifier <evaluation.json> <negative-controls.json> <evaluation.schema.json> <negative-controls.schema.json>"
        );
        return ExitCode::from(2);
    }
    let bundle = match read_json(&args[1]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let controls = match read_json(&args[2]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let evaluation_schema = match read_json(&args[3]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };
    let controls_schema = match read_json(&args[4]) {
        Ok(value) => value,
        Err(error) => { eprintln!("{error}"); return ExitCode::FAILURE; }
    };

    let mut findings =
        validate_schema_pair(&bundle, &evaluation_schema, &controls, &controls_schema);
    findings.extend(validate_bundle(&bundle));

    let fixtures = controls["fixtures"].as_array().cloned().unwrap_or_default();
    if fixtures.len() != EXPECTED_FIXTURES
        || controls["fixture_count"].as_u64() != Some(EXPECTED_FIXTURES as u64)
    {
        findings.push("negative_control_fixture_count_invalid".to_string());
    }
    for (index, fixture) in fixtures.iter().enumerate() {
        let mut changed = bundle.clone();
        if !apply_fixture(&mut changed, fixture) {
            findings.push(format!("negative_control_fixture_{index}_mutation_invalid"));
            continue;
        }
        let actual = validate_bundle(&changed);
        let expected = string_set(&fixture["expected_violations"]);
        let actual: BTreeSet<String> = actual.into_iter().collect();
        if expected != actual {
            findings.push(format!(
                "negative_control_fixture_{}_oracle_mismatch",
                fixture["fixture_id"].as_str().unwrap_or("unknown")
            ));
        }
    }
    if !findings.is_empty() {
        findings.sort();
        findings.dedup();
        eprintln!("CONFIDENCE CALIBRATION QUALIFICATION FAILED:");
        for finding in findings { eprintln!(" - {finding}"); }
        return ExitCode::FAILURE;
    }
    let Some(metrics) = calculate_metrics(&bundle) else {
        eprintln!("metrics are unavailable");
        return ExitCode::FAILURE;
    };
    println!(
        concat!(
            "QUALIFIED STRUCTURAL CONFIDENCE EVALUATOR: n={} ",
            "independent_groups={} uncertainty_interval={}"
        ),
        metrics.sample_count,
        metrics.independent_group_count,
        metrics.uncertainty_interval_status
    );
    println!(
        "Brier={:.8} log_loss={:.8} ECE10={:.8} tie_aware_AURC={:.8}",
        metrics.brier_score, metrics.log_loss, metrics.ece_equal_width, metrics.aurc
    );
    for (coverage, risk) in metrics.risk_at_coverage {
        println!("risk@coverage[{:.0}%]={:.8}", coverage * 100.0, risk);
    }
    println!("NOT A CALIBRATOR: no calibration map was fitted by this evaluator.");
    println!(
        "NOT AUTHORITY: confidence does not create rights, voice, vote weight, or authority."
    );
    println!("SYNTHETIC ONLY: the checked-in outcomes are fictional known-answer controls.");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUNDLE: &str = include_str!("../../CONFIDENCE_EVALUATION_V0_1.json");
    const CONTROLS: &str = include_str!("../../CONFIDENCE_NEGATIVE_CONTROLS_V0_1.json");
    const BUNDLE_SCHEMA: &str =
        include_str!("../../CONFIDENCE_EVALUATION_V0_1.schema.json");
    const CONTROLS_SCHEMA: &str =
        include_str!("../../CONFIDENCE_NEGATIVE_CONTROLS_V0_1.schema.json");

    fn parsed(source: &str) -> Value {
        serde_json::from_str(source).expect("checked-in JSON must parse")
    }

    #[test]
    fn checked_in_schemas_are_applied_to_both_artifacts() {
        let errors = validate_schema_pair(
            &parsed(BUNDLE),
            &parsed(BUNDLE_SCHEMA),
            &parsed(CONTROLS),
            &parsed(CONTROLS_SCHEMA),
        );
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn unsupported_keyword_in_absent_optional_schema_property_is_rejected() {
        let mut schema = parsed(BUNDLE_SCHEMA);
        schema["properties"]["unused_optional"] =
            serde_json::json!({"type":"string","maxLength":10});
        let errors = validate_schema_pair(
            &parsed(BUNDLE),
            &schema,
            &parsed(CONTROLS),
            &parsed(CONTROLS_SCHEMA),
        );
        assert!(errors.iter().any(|error| error.contains("unsupported_schema_keyword:maxLength")));
    }

    #[test]
    fn evaluation_artifact_rejects_undeclared_properties() {
        let mut bundle = parsed(BUNDLE);
        bundle["undeclared_field"] = Value::Bool(true);
        let errors = validate_schema_pair(
            &bundle,
            &parsed(BUNDLE_SCHEMA),
            &parsed(CONTROLS),
            &parsed(CONTROLS_SCHEMA),
        );
        assert!(errors.iter().any(|error| error.contains("schema_additional_property:undeclared_field")));
    }

    #[test]
    fn negative_control_schema_rejects_undeclared_fixture_fields() {
        let mut controls = parsed(CONTROLS);
        controls["fixtures"][0]["unregistered"] = Value::Bool(true);
        let errors = validate_schema_pair(
            &parsed(BUNDLE),
            &parsed(BUNDLE_SCHEMA),
            &controls,
            &parsed(CONTROLS_SCHEMA),
        );
        assert!(errors.iter().any(|error| error.contains("schema_additional_property:unregistered")));
    }

    #[test]
    fn checked_in_synthetic_evaluation_is_structurally_valid() {
        assert!(validate_bundle(&parsed(BUNDLE)).is_empty());
    }

    #[test]
    fn proper_scores_match_known_answer_fixture() {
        let metrics = calculate_metrics(&parsed(BUNDLE)).unwrap();
        assert_eq!(metrics.sample_count, 50);
        assert!((metrics.brier_score - 0.16).abs() < 1e-12);
        assert!(metrics.ece_equal_width.abs() < 1e-12);
        assert!((metrics.log_loss - 0.5004024235381879).abs() < 1e-12);
        assert_eq!(metrics.reliability_bins.len(), 2);
        assert!((metrics.aurc - 0.29502585182722457).abs() < 1e-12);
        assert_eq!(metrics.independent_group_count, 2);
        assert_eq!(metrics.uncertainty_interval_status, "NotEstimated");
        assert_eq!(
            metrics.risk_at_coverage,
            vec![(0.25, 0.2), (0.50, 0.2), (0.75, 0.4), (1.00, 0.5)]
        );
    }

    #[test]
    fn tied_confidence_risk_metrics_are_invariant_to_row_order() {
        let original = parsed(BUNDLE);
        let mut reordered = original.clone();
        reordered["observations"].as_array_mut().unwrap().reverse();
        let original_metrics = calculate_metrics(&original).unwrap();
        let reordered_metrics = calculate_metrics(&reordered).unwrap();
        assert!((original_metrics.aurc - reordered_metrics.aurc).abs() < 1e-12);
        assert_eq!(
            original_metrics.risk_at_coverage.len(),
            reordered_metrics.risk_at_coverage.len()
        );
        for (left, right) in original_metrics
            .risk_at_coverage
            .iter()
            .zip(reordered_metrics.risk_at_coverage.iter())
        {
            assert!((left.0 - right.0).abs() < 1e-12);
            assert!((left.1 - right.1).abs() < 1e-12);
        }
    }

    #[test]
    fn cluster_uncertainty_is_not_claimed_from_two_synthetic_groups() {
        let metrics = calculate_metrics(&parsed(BUNDLE)).unwrap();
        assert_eq!(metrics.independent_group_count, 2);
        assert_eq!(metrics.uncertainty_interval_status, "NotEstimated");
    }

    #[test]
    fn high_confidence_subset_has_lower_selective_risk() {
        let metrics = calculate_metrics(&parsed(BUNDLE)).unwrap();
        assert!(metrics.risk_at_coverage[0].1 < metrics.risk_at_coverage[3].1);
    }

    #[test]
    fn all_twenty_four_negative_controls_match_expected_violation_codes() {
        let bundle = parsed(BUNDLE);
        let controls = parsed(CONTROLS);
        let fixtures = controls["fixtures"].as_array().unwrap();
        assert_eq!(fixtures.len(), EXPECTED_FIXTURES);
        for fixture in fixtures {
            let mut changed = bundle.clone();
            assert!(apply_fixture(&mut changed, fixture));
            let actual: BTreeSet<String> = validate_bundle(&changed).into_iter().collect();
            let expected = string_set(&fixture["expected_violations"]);
            assert_eq!(actual, expected, "fixture {}", field(fixture, "fixture_id").unwrap());
        }
    }

    #[test]
    fn normative_values_cannot_be_scored_as_binary_correctness() {
        let mut bundle = parsed(BUNDLE);
        bundle["evaluation_identity"]["task_truth_profile"] =
            Value::String("NormativePluralism".into());
        assert!(
            validate_bundle(&bundle).contains(&"evaluation_truth_profile_unsupported".to_string())
        );
    }

    #[test]
    fn confidence_is_not_a_civic_authority_field() {
        let profile = &parsed(BUNDLE)["calibration_profile"];
        assert!(profile.get("vote_weight").is_none());
        assert!(profile.get("eligibility").is_none());
        assert!(profile.get("authority").is_none());
    }
}
