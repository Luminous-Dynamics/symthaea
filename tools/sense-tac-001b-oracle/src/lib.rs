//! Independent reference oracle for SENSE-TAC-001B.
//!
//! The fixture corpus is synthetic and dimensionless (arb). This crate
//! validates the closed fixture schema, recomputes declared deterministic
//! signal transforms, checks fault-specific evidence dispositions, and guards
//! against representation aliases and false independence.
//!
//! It does not model a specific physical tactile sensor, certify calibration,
//! estimate real fault rates, or establish physical task performance.

#![forbid(unsafe_code)]

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use std::fmt;

pub const CORPUS_JSON: &str =
    include_str!("../../../docs/research/fixtures/SENSE_TAC_001B_FAULT_CORPUS_V0_1.json");
pub const CORPUS_SCHEMA_JSON: &str =
    include_str!("../../../docs/research/fixtures/SENSE_TAC_001B_FAULT_CORPUS_V0_1.schema.json");

pub const EXPECTED_CORPUS_BLOB_SHA1: &str = "e9349751590c4a3a28f4be81a5a11f8170ef2212";
pub const EXPECTED_SCHEMA_BLOB_SHA1: &str = "7b531325c1c8f105879dd80e0ba6016555a4b645";

const EXPECTED_SCHEMA_ID: &str =
    "https://luminousdynamics.io/schemas/sense-tac-001b-fault-corpus-v0.1.schema.json";
const EXPECTED_CORPUS_ID: &str = "SENSE-TAC-001B-FAULT-CORPUS";
const EXPECTED_VERSION: &str = "0.1.0";
const EXPECTED_CASE_COUNT: usize = 30;
const NUMERIC_TOLERANCE: f64 = 1.0e-9;

const EXPECTED_CASES: &[(&str, &str, &str)] = &[
    ("TAC-NOM-001", "nominal", "none"),
    ("TAC-BIAS-002", "static_error", "bias"),
    ("TAC-GAIN-003", "static_error", "gain"),
    ("TAC-XTALK-004", "cross_axis", "normal_to_shear_crosstalk"),
    ("TAC-XTALK-005", "cross_axis", "shear_to_normal_crosstalk"),
    ("TAC-SAT-006", "range_fault", "saturation"),
    ("TAC-DEAD-007", "range_fault", "dead_zone"),
    ("TAC-QUANT-008", "quantization", "quantization"),
    ("TAC-DRIFT-009", "time_dependent_error", "linear_drift"),
    (
        "TAC-TEMP-010",
        "environmental_error",
        "temperature_coefficient",
    ),
    ("TAC-HYST-011", "history_dependence", "hysteresis"),
    ("TAC-DROP-012", "missingness", "dropout"),
    ("TAC-STUCK-013", "missingness", "stuck_at"),
    ("TAC-OPEN-014", "missingness", "intermittent_open"),
    ("TAC-SWAP-015", "identity", "channel_swap"),
    ("TAC-UNIT-016", "units", "unit_scale_mismatch"),
    ("TAC-CLOCK-017", "timing", "clock_offset"),
    ("TAC-LAT-018", "timing", "readout_latency"),
    ("TAC-JITTER-019", "timing", "timestamp_jitter"),
    ("TAC-DUP-020", "replay", "exact_duplicate"),
    ("TAC-CONFLICT-021", "replay", "conflicting_duplicate"),
    ("TAC-ADC-COMMON-022", "common_mode", "shared_adc_offset"),
    ("TAC-REF-GAIN-023", "common_mode", "shared_reference_gain"),
    (
        "TAC-ARTICLE-024",
        "calibration",
        "wrong_article_calibration",
    ),
    ("TAC-STALE-025", "calibration", "firmware_generation_change"),
    ("TAC-REMOUNT-026", "installation", "remount_offset"),
    (
        "TAC-CURVE-027",
        "applicability",
        "curvature_profile_mismatch",
    ),
    (
        "TAC-SELFTEST-028",
        "self_test",
        "self_test_misses_scale_drift",
    ),
    (
        "TAC-HDC-029",
        "representation_loss",
        "representation_threshold_loss",
    ),
    (
        "TAC-ALIAS-030",
        "representation_alias",
        "representation_alias",
    ),
];

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationReport {
    pub fixture_count: usize,
    pub unique_fault_kinds: usize,
    pub cases_verified: usize,
}

#[derive(Debug)]
struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueVisitor;

        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueJson;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value with no duplicate object members")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueJson(Value::Number(number)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::String(value)))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::Null))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(UniqueJson(Value::Null))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(UniqueJson(value)) = sequence.next_element::<UniqueJson>()? {
                    values.push(value);
                }
                Ok(UniqueJson(Value::Array(values)))
            }

            fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = Map::new();
                while let Some(key) = object.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON object member: {key}"
                        )));
                    }
                    let UniqueJson(value) = object.next_value::<UniqueJson>()?;
                    values.insert(key, value);
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }

        deserializer.deserialize_any(UniqueVisitor)
    }
}

fn parse_unique(raw: &str, label: &str) -> Result<Value, String> {
    serde_json::from_str::<UniqueJson>(raw)
        .map(|UniqueJson(value)| value)
        .map_err(|error| format!("{label}: {error}"))
}

fn path_error(path: &str, message: impl AsRef<str>) -> String {
    format!("{path}: {}", message.as_ref())
}

fn schema_type_matches(value: &Value, expected_type: &Value) -> bool {
    let matches_one = |type_name: &str| match type_name {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    };

    match expected_type {
        Value::String(name) => matches_one(name),
        Value::Array(names) => names.iter().filter_map(Value::as_str).any(matches_one),
        _ => false,
    }
}

/// Small JSON Schema evaluator for the closed v0.1 schema vocabulary.
/// Unsupported schema keywords fail closed instead of being ignored.
fn validate_schema(instance: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    let Some(schema_object) = schema.as_object() else {
        errors.push(path_error(path, "schema node is not an object"));
        return;
    };

    const SUPPORTED: &[&str] = &[
        "$schema",
        "$id",
        "title",
        "type",
        "additionalProperties",
        "required",
        "properties",
        "items",
        "enum",
        "const",
        "minItems",
        "maxItems",
        "minLength",
        "minProperties",
        "pattern",
    ];
    for keyword in schema_object.keys() {
        if !SUPPORTED.contains(&keyword.as_str()) {
            errors.push(path_error(
                path,
                format!("unsupported schema keyword {keyword}"),
            ));
        }
    }

    if let Some(expected) = schema_object.get("const")
        && instance != expected
    {
        errors.push(path_error(path, format!("expected const {expected}")));
    }

    if let Some(values) = schema_object.get("enum").and_then(Value::as_array)
        && !values.iter().any(|expected| expected == instance)
    {
        errors.push(path_error(path, "value is not in the declared enum"));
    }

    if let Some(expected_type) = schema_object.get("type")
        && !schema_type_matches(instance, expected_type)
    {
        errors.push(path_error(
            path,
            format!("type mismatch; expected {expected_type}"),
        ));
        return;
    }

    if let Some(object) = instance.as_object() {
        if let Some(required) = schema_object.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(key) {
                    errors.push(path_error(path, format!("missing required member {key}")));
                }
            }
        }

        if let Some(minimum) = schema_object.get("minProperties").and_then(Value::as_u64)
            && object.len() < minimum as usize
        {
            errors.push(path_error(
                path,
                format!("requires at least {minimum} properties"),
            ));
        }

        let properties = schema_object.get("properties").and_then(Value::as_object);

        for (key, value) in object {
            if let Some(child_schema) = properties.and_then(|items| items.get(key)) {
                validate_schema(value, child_schema, &format!("{path}.{key}"), errors);
            } else if schema_object.get("additionalProperties") == Some(&Value::Bool(false)) {
                errors.push(path_error(path, format!("unknown member {key}")));
            } else if let Some(extra_schema) = schema_object
                .get("additionalProperties")
                .filter(|value| value.is_object())
            {
                validate_schema(value, extra_schema, &format!("{path}.{key}"), errors);
            }
        }
    }

    if let Some(array) = instance.as_array() {
        if let Some(minimum) = schema_object.get("minItems").and_then(Value::as_u64)
            && array.len() < minimum as usize
        {
            errors.push(path_error(
                path,
                format!("requires at least {minimum} items"),
            ));
        }
        if let Some(maximum) = schema_object.get("maxItems").and_then(Value::as_u64)
            && array.len() > maximum as usize
        {
            errors.push(path_error(path, format!("allows at most {maximum} items")));
        }
        if let Some(item_schema) = schema_object.get("items") {
            for (index, value) in array.iter().enumerate() {
                validate_schema(value, item_schema, &format!("{path}[{index}]"), errors);
            }
        }
    }

    if let Some(string) = instance.as_str() {
        if let Some(minimum) = schema_object.get("minLength").and_then(Value::as_u64)
            && string.chars().count() < minimum as usize
        {
            errors.push(path_error(
                path,
                format!("requires at least {minimum} characters"),
            ));
        }
        if schema_object.get("pattern").and_then(Value::as_str) == Some("^TAC-[A-Z0-9-]+$") {
            let suffix = string.strip_prefix("TAC-").unwrap_or("");
            if suffix.is_empty()
                || !suffix.chars().all(|character| {
                    character.is_ascii_uppercase() || character.is_ascii_digit() || character == '-'
                })
            {
                errors.push(path_error(
                    path,
                    "case identifier does not match TAC pattern",
                ));
            }
        }
    }
}

fn member<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| path_error(path, format!("missing {key}")))
}

fn string_member<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    member(value, key, path)?
        .as_str()
        .ok_or_else(|| path_error(path, format!("{key} must be a string")))
}

fn number_member(value: &Value, key: &str, path: &str) -> Result<f64, String> {
    member(value, key, path)?
        .as_f64()
        .ok_or_else(|| path_error(path, format!("{key} must be a number")))
}

fn optional_number(value: &Value, key: &str, path: &str) -> Result<Option<f64>, String> {
    let raw = member(value, key, path)?;
    if raw.is_null() {
        Ok(None)
    } else {
        raw.as_f64()
            .map(Some)
            .ok_or_else(|| path_error(path, format!("{key} must be a number or null")))
    }
}

fn parameter_number(fault: &Value, path: &str) -> Result<f64, String> {
    member(fault, "parameter", path)?
        .as_f64()
        .ok_or_else(|| path_error(path, "fault parameter must be numeric for this fault"))
}

fn nearly_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= NUMERIC_TOLERANCE
}

fn require_near(
    actual: Option<f64>,
    expected: f64,
    case_id: &str,
    field: &str,
) -> Result<(), String> {
    match actual {
        Some(value) if nearly_equal(value, expected) => Ok(()),
        _ => Err(path_error(
            case_id,
            format!("{field} does not match the deterministic model"),
        )),
    }
}

fn require_string(value: &Value, key: &str, expected: &str, path: &str) -> Result<(), String> {
    let actual = string_member(value, key, path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(path_error(
            path,
            format!("{key}: expected {expected}, got {actual}"),
        ))
    }
}

fn expected_for_fault(kind: &str) -> Result<Value, String> {
    let (evidence, flags, task, independence, representation): (&str, &[&str], &str, &str, &str) =
        match kind {
            "none" => (
                "eligible",
                &[],
                "usable",
                "one_physical_root",
                "raw_preserved",
            ),
            "bias" => (
                "limited",
                &["bias_suspect"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "gain" => (
                "limited",
                &["gain_suspect"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "normal_to_shear_crosstalk" | "shear_to_normal_crosstalk" => (
                "limited",
                &["cross_axis_coupling"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "saturation" => (
                "reject",
                &["saturated"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "dead_zone" => (
                "limited",
                &["dead_zone"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "quantization" => (
                "limited",
                &["quantized"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "linear_drift" => (
                "limited",
                &["drift_suspect"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "temperature_coefficient" => (
                "limited",
                &["environmental_compensation_unresolved"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "hysteresis" => (
                "limited",
                &["history_dependent"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "dropout" => (
                "reject",
                &["missing"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "stuck_at" => (
                "reject",
                &["stuck_at"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "intermittent_open" => (
                "limited",
                &["intermittent_open"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "channel_swap" => (
                "reject",
                &["channel_identity_mismatch"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "unit_scale_mismatch" => (
                "reject",
                &["unit_mismatch"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "clock_offset" => (
                "unknown",
                &["clock_offset"],
                "indeterminate",
                "one_physical_root",
                "raw_preserved",
            ),
            "readout_latency" => (
                "limited",
                &["readout_latency"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "timestamp_jitter" => (
                "limited",
                &["timestamp_jitter"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "exact_duplicate" => (
                "idempotent_duplicate",
                &["duplicate"],
                "usable_once_only",
                "one_physical_root",
                "raw_preserved",
            ),
            "conflicting_duplicate" => (
                "conflict",
                &["conflict"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "shared_adc_offset" | "shared_reference_gain" => (
                "limited",
                &["common_mode"],
                "degraded",
                "shared_common_mode",
                "raw_preserved",
            ),
            "wrong_article_calibration" => (
                "reject",
                &["calibration_article_mismatch"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "firmware_generation_change" => (
                "reject",
                &["stale_calibration"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "remount_offset" => (
                "unknown",
                &["remount_mismatch"],
                "indeterminate",
                "one_physical_root",
                "raw_preserved",
            ),
            "curvature_profile_mismatch" => (
                "out_of_profile",
                &["out_of_profile"],
                "abstain",
                "one_physical_root",
                "raw_preserved",
            ),
            "self_test_misses_scale_drift" => (
                "limited",
                &["self_test_insufficient", "gain_suspect"],
                "degraded",
                "one_physical_root",
                "raw_preserved",
            ),
            "representation_threshold_loss" => (
                "eligible",
                &["representation_loss"],
                "abstain",
                "not_independent_descendant",
                "lossy",
            ),
            "representation_alias" => (
                "eligible",
                &["representation_alias"],
                "indeterminate",
                "not_independent_descendant",
                "alias",
            ),
            _ => return Err(format!("unsupported fault kind {kind}")),
        };

    Ok(json!({
        "evidence_disposition": evidence,
        "qc_flags": flags,
        "task_disposition": task,
        "independence_disposition": independence,
        "representation_disposition": representation,
    }))
}

fn require_shared_roots(input: &Value, path: &str) -> Result<(), String> {
    let roots = member(input, "source_roots", path)?
        .as_array()
        .ok_or_else(|| path_error(path, "source_roots must be an array"))?;
    let has_adc = roots
        .iter()
        .any(|value| value.as_str() == Some("adc-shared"));
    let has_reference = roots
        .iter()
        .any(|value| value.as_str() == Some("ref-shared"));
    if has_adc && has_reference {
        Ok(())
    } else {
        Err(path_error(
            path,
            "shared ADC/reference common-mode roots are not explicit",
        ))
    }
}

fn check_numeric_model(case: &Value, model: &Value) -> Result<(), String> {
    let case_id = string_member(case, "id", "case")?;
    let input = member(case, "candidate_input", case_id)?;
    let evaluator = member(case, "evaluator_only", case_id)?;
    let truth = member(evaluator, "physical_truth", case_id)?;
    let fault = member(evaluator, "fault", case_id)?;
    let kind = string_member(fault, "kind", case_id)?;
    let normal = optional_number(input, "normal_raw", case_id)?;
    let shear = optional_number(input, "shear_raw", case_id)?;
    let truth_normal = number_member(truth, "normal", case_id)?;
    let truth_shear = number_member(truth, "shear", case_id)?;
    let truth_temperature = number_member(truth, "temperature_c", case_id)?;
    let parameter = member(fault, "parameter", case_id)?;
    let reference_temperature = number_member(model, "reference_temperature_c", "model")?;
    let drift_epoch = number_member(model, "drift_epoch_ms", "model")?;

    match kind {
        "none" => {
            let gain = number_member(model, "default_channel_gain", "model")?;
            let bias = number_member(model, "default_channel_bias", "model")?;
            require_near(normal, truth_normal * gain + bias, case_id, "normal_raw")?;
            require_near(shear, truth_shear * gain + bias, case_id, "shear_raw")?;
        }
        "bias" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, truth_normal + p, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "gain" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, truth_normal * p, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "normal_to_shear_crosstalk" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, truth_normal, case_id, "normal_raw")?;
            require_near(shear, truth_shear + p * truth_normal, case_id, "shear_raw")?;
        }
        "shear_to_normal_crosstalk" => {
            let p = parameter_number(fault, case_id)?;
            require_near(
                normal,
                truth_normal + p * truth_shear,
                case_id,
                "normal_raw",
            )?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "saturation" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, truth_normal.min(p), case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "dead_zone" => {
            let p = parameter_number(fault, case_id)?;
            let expected = if truth_normal.abs() < p {
                0.0
            } else {
                truth_normal
            };
            require_near(normal, expected, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "quantization" => {
            let p = parameter_number(fault, case_id)?;
            if p <= 0.0 {
                return Err(path_error(case_id, "quantization step must be positive"));
            }
            let quantize = |value: f64| value.signum() * ((value.abs() / p + 0.5).floor()) * p;
            require_near(normal, quantize(truth_normal), case_id, "normal_raw")?;
            require_near(shear, quantize(truth_shear), case_id, "shear_raw")?;
        }
        "linear_drift" => {
            let p = parameter_number(fault, case_id)?;
            let time_ms = number_member(input, "time_ms", case_id)?;
            require_near(
                normal,
                truth_normal + p * (time_ms - drift_epoch),
                case_id,
                "normal_raw",
            )?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "temperature_coefficient" => {
            let p = parameter_number(fault, case_id)?;
            require_near(
                normal,
                truth_normal + p * (truth_temperature - reference_temperature),
                case_id,
                "normal_raw",
            )?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "hysteresis" => {
            let p = parameter_number(fault, case_id)?;
            require_string(truth, "event", "unloading", case_id)?;
            require_near(normal, truth_normal + p, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "dropout" => {
            require_string(input, "sample_presence", "missing", case_id)?;
            if normal.is_some() || shear.is_some() {
                return Err(path_error(
                    case_id,
                    "dropout must preserve missing values as null",
                ));
            }
        }
        "stuck_at" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, p, case_id, "normal_raw")?;
            if nearly_equal(p, truth_normal) {
                return Err(path_error(
                    case_id,
                    "stuck-at value must differ from physical truth",
                ));
            }
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "intermittent_open" => {
            require_string(input, "sample_presence", "intermittent_open", case_id)?;
            if normal.is_some() {
                return Err(path_error(
                    case_id,
                    "opened normal channel must remain null",
                ));
            }
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "channel_swap" => {
            let order = member(input, "channel_order", case_id)?
                .as_array()
                .ok_or_else(|| path_error(case_id, "channel_order must be an array"))?;
            if order.len() != 2
                || order[0].as_str() != Some("shear")
                || order[1].as_str() != Some("normal")
            {
                return Err(path_error(
                    case_id,
                    "channel swap identity is not represented",
                ));
            }
            require_near(normal, truth_shear, case_id, "normal_raw")?;
            require_near(shear, truth_normal, case_id, "shear_raw")?;
        }
        "unit_scale_mismatch" => {
            let p = parameter_number(fault, case_id)?;
            require_string(input, "reported_unit", "milli-arb", case_id)?;
            require_near(normal, truth_normal * p, case_id, "normal_raw")?;
            require_near(shear, truth_shear * p, case_id, "shear_raw")?;
        }
        "clock_offset" => {
            let p = parameter_number(fault, case_id)?;
            let device_time = number_member(input, "time_ms", case_id)?;
            let reference_time = number_member(truth, "reference_time_ms", case_id)?;
            if !nearly_equal(device_time - reference_time, p) {
                return Err(path_error(
                    case_id,
                    "clock offset does not match independent reference",
                ));
            }
            require_near(normal, truth_normal, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "readout_latency" => {
            let p = parameter_number(fault, case_id)?;
            let source_time = number_member(input, "time_ms", case_id)?;
            let receipt_time = number_member(input, "receipt_time_ms", case_id)?;
            if !nearly_equal(receipt_time - source_time, p) {
                return Err(path_error(
                    case_id,
                    "readout latency does not match source/receipt times",
                ));
            }
            require_near(normal, truth_normal, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "timestamp_jitter" => {
            let p = parameter_number(fault, case_id)?;
            let device_time = number_member(input, "time_ms", case_id)?;
            let reference_time = number_member(truth, "reference_time_ms", case_id)?;
            if !nearly_equal(device_time - reference_time, p) {
                return Err(path_error(
                    case_id,
                    "timestamp residual does not match reference",
                ));
            }
            require_near(normal, truth_normal, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "exact_duplicate" | "conflicting_duplicate" => {
            let replay = member(input, "replay_context", case_id)?;
            let current_sequence = number_member(input, "sequence", case_id)?;
            let previous_sequence = number_member(replay, "previous_sequence", case_id)?;
            if !nearly_equal(current_sequence, previous_sequence) {
                return Err(path_error(
                    case_id,
                    "replay fixture must reuse the exact sequence",
                ));
            }
            let current_payload = string_member(input, "payload_identity", case_id)?;
            let previous_payload = string_member(replay, "previous_payload_identity", case_id)?;
            if kind == "exact_duplicate" {
                if current_payload != previous_payload {
                    return Err(path_error(
                        case_id,
                        "exact replay must preserve payload identity",
                    ));
                }
                require_string(replay, "expected_relation", "same_payload", case_id)?;
                require_near(normal, truth_normal, case_id, "normal_raw")?;
                require_near(shear, truth_shear, case_id, "shear_raw")?;
            } else {
                if current_payload == previous_payload {
                    return Err(path_error(
                        case_id,
                        "conflicting replay must change payload identity",
                    ));
                }
                require_string(replay, "expected_relation", "different_payload", case_id)?;
            }
        }
        "shared_adc_offset" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, truth_normal + p, case_id, "normal_raw")?;
            require_near(shear, truth_shear + p, case_id, "shear_raw")?;
            require_shared_roots(input, case_id)?;
        }
        "shared_reference_gain" => {
            let p = parameter_number(fault, case_id)?;
            require_near(normal, truth_normal * p, case_id, "normal_raw")?;
            require_near(shear, truth_shear * p, case_id, "shear_raw")?;
            require_shared_roots(input, case_id)?;
        }
        "wrong_article_calibration" => {
            let physical_article = string_member(truth, "physical_article", case_id)?;
            let calibration_article = parameter
                .as_str()
                .ok_or_else(|| path_error(case_id, "expected calibration article label"))?;
            let calibration_ref = string_member(input, "calibration_ref", case_id)?;
            if physical_article == calibration_article
                || !calibration_ref.contains(calibration_article)
            {
                return Err(path_error(
                    case_id,
                    "calibration/article mismatch is not explicit",
                ));
            }
            require_near(normal, truth_normal, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "firmware_generation_change" => {
            require_string(input, "firmware_generation", "fw-b", case_id)?;
            let calibration_ref = string_member(input, "calibration_ref", case_id)?;
            if !calibration_ref.contains("fw-a") {
                return Err(path_error(
                    case_id,
                    "calibration does not identify the old firmware generation",
                ));
            }
            require_near(normal, truth_normal, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "remount_offset" => {
            let p = parameter_number(fault, case_id)?;
            require_string(input, "installation_ref", "mount-b", case_id)?;
            require_near(normal, truth_normal + p, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "curvature_profile_mismatch" => {
            require_string(truth, "mount_profile", "curved", case_id)?;
            require_string(input, "calibration_ref", "cal-flat", case_id)?;
            require_string(input, "installation_ref", "mount-curved", case_id)?;
        }
        "self_test_misses_scale_drift" => {
            let p = parameter_number(fault, case_id)?;
            require_string(input, "self_test", "pass", case_id)?;
            require_near(normal, truth_normal * p, case_id, "normal_raw")?;
            require_near(shear, truth_shear, case_id, "shear_raw")?;
        }
        "representation_threshold_loss" => {
            let p = parameter_number(fault, case_id)?;
            let raw_normal =
                normal.ok_or_else(|| path_error(case_id, "raw normal observation missing"))?;
            let output = number_member(input, "representation_output", case_id)?;
            if raw_normal >= p || !nearly_equal(output, 0.0) {
                return Err(path_error(
                    case_id,
                    "threshold-loss representation does not erase the declared cue",
                ));
            }
            if string_member(input, "representation_id", case_id)? == "raw" {
                return Err(path_error(
                    case_id,
                    "lossy representation must retain a distinct identity",
                ));
            }
        }
        "representation_alias" => {
            let comparison = member(input, "representation_comparison", case_id)?;
            let comparison_normal = number_member(comparison, "comparison_normal_raw", case_id)?;
            let comparison_shear = number_member(comparison, "comparison_shear_raw", case_id)?;
            let comparison_representation =
                string_member(comparison, "comparison_representation_id", case_id)?;
            let representation = string_member(input, "representation_id", case_id)?;
            let raw_normal =
                normal.ok_or_else(|| path_error(case_id, "primary raw normal missing"))?;
            let raw_shear =
                shear.ok_or_else(|| path_error(case_id, "primary raw shear missing"))?;
            if nearly_equal(comparison_normal, raw_normal)
                || !nearly_equal(comparison_shear, raw_shear)
                || comparison_representation != representation
            {
                return Err(path_error(
                    case_id,
                    "two evaluator-distinct traces do not alias to one representation",
                ));
            }
        }
        _ => return Err(path_error(case_id, "unhandled fault kind")),
    }

    Ok(())
}

fn validate_model(model: &Value) -> Result<(), String> {
    let reference_temperature = number_member(model, "reference_temperature_c", "model")?;
    let drift_epoch = number_member(model, "drift_epoch_ms", "model")?;
    let default_gain = number_member(model, "default_channel_gain", "model")?;
    let default_bias = number_member(model, "default_channel_bias", "model")?;

    if !nearly_equal(reference_temperature, 20.0)
        || !nearly_equal(drift_epoch, 0.0)
        || !nearly_equal(default_gain, 1.0)
        || !nearly_equal(default_bias, 0.0)
    {
        return Err(
            "model reference constants changed without a new fixture generation".to_owned(),
        );
    }
    require_string(
        model,
        "drift_elapsed_time_basis",
        "time_ms - drift_epoch_ms",
        "model",
    )?;

    let parameter_semantics = member(model, "parameter_semantics", "model")?
        .as_object()
        .ok_or_else(|| "model.parameter_semantics must be an object".to_owned())?;
    for required in [
        "bias",
        "gain",
        "normal_to_shear_crosstalk",
        "shear_to_normal_crosstalk",
        "saturation",
        "dead_zone",
        "quantization",
        "linear_drift",
        "temperature_coefficient",
        "hysteresis",
        "shared_adc_offset",
        "shared_reference_gain",
        "clock_offset",
        "readout_latency",
        "timestamp_jitter",
    ] {
        if !parameter_semantics.contains_key(required) {
            return Err(format!("model.parameter_semantics missing {required}"));
        }
    }
    Ok(())
}

pub fn validate_document(document: &Value, schema: &Value) -> Result<ValidationReport, String> {
    if string_member(document, "schema_id", "corpus")? != EXPECTED_SCHEMA_ID {
        return Err("unexpected corpus schema_id".to_owned());
    }
    if string_member(document, "corpus_id", "corpus")? != EXPECTED_CORPUS_ID {
        return Err("unexpected corpus_id".to_owned());
    }
    if string_member(document, "version", "corpus")? != EXPECTED_VERSION {
        return Err("unexpected corpus version".to_owned());
    }
    require_string(document, "status", "synthetic_reference_only", "corpus")?;

    let mut schema_errors = Vec::new();
    validate_schema(document, schema, "$", &mut schema_errors);
    if !schema_errors.is_empty() {
        return Err(format!(
            "schema validation failed: {}",
            schema_errors.join("; ")
        ));
    }

    let model = member(document, "model", "corpus")?;
    validate_model(model)?;

    let cases = member(document, "cases", "corpus")?
        .as_array()
        .ok_or_else(|| "corpus.cases must be an array".to_owned())?;
    if cases.len() != EXPECTED_CASE_COUNT || cases.len() != EXPECTED_CASES.len() {
        return Err(format!(
            "expected {EXPECTED_CASE_COUNT} fixtures, found {}",
            cases.len()
        ));
    }

    let mut identifiers = BTreeSet::new();
    let mut fault_kinds = BTreeSet::new();

    for (index, case) in cases.iter().enumerate() {
        let case_id = string_member(case, "id", "case")?;
        let family = string_member(case, "family", case_id)?;
        let fault = member(member(case, "evaluator_only", case_id)?, "fault", case_id)?;
        let kind = string_member(fault, "kind", case_id)?;
        let (expected_id, expected_family, expected_kind) = EXPECTED_CASES[index];

        if case_id != expected_id || family != expected_family || kind != expected_kind {
            return Err(path_error(
                case_id,
                format!(
                    "fixture identity drift: expected ({expected_id}, {expected_family}, {expected_kind})"
                ),
            ));
        }
        if !identifiers.insert(case_id.to_owned()) {
            return Err(path_error(case_id, "duplicate fixture identity"));
        }

        let derived_expected = expected_for_fault(kind)?;
        let recorded_expected = member(
            member(case, "evaluator_only", case_id)?,
            "expected",
            case_id,
        )?;
        if &derived_expected != recorded_expected {
            return Err(path_error(
                case_id,
                format!(
                    "expected disposition drift: derived {derived_expected}, recorded {recorded_expected}"
                ),
            ));
        }

        check_numeric_model(case, model)?;
        fault_kinds.insert(kind.to_owned());
    }

    Ok(ValidationReport {
        fixture_count: cases.len(),
        unique_fault_kinds: fault_kinds.len(),
        cases_verified: cases.len(),
    })
}

pub fn validate_tracked() -> Result<ValidationReport, String> {
    let document = parse_unique(CORPUS_JSON, "tracked corpus")?;
    let schema = parse_unique(CORPUS_SCHEMA_JSON, "tracked corpus schema")?;
    validate_document(&document, &schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracked_values() -> (Value, Value) {
        let document = parse_unique(CORPUS_JSON, "test corpus").expect("valid tracked corpus");
        let schema = parse_unique(CORPUS_SCHEMA_JSON, "test schema").expect("valid tracked schema");
        (document, schema)
    }

    fn case_mut<'a>(document: &'a mut Value, case_id: &str) -> &'a mut Value {
        document["cases"]
            .as_array_mut()
            .expect("cases array")
            .iter_mut()
            .find(|case| case["id"].as_str() == Some(case_id))
            .expect("fixture exists")
    }

    #[test]
    fn tracked_corpus_matches_all_thirty_known_answers() {
        let report = validate_tracked().expect("tracked corpus is valid");
        assert_eq!(report.fixture_count, 30);
        assert_eq!(report.unique_fault_kinds, 30);
        assert_eq!(report.cases_verified, 30);
    }

    #[test]
    fn saturated_reading_cannot_become_valid_by_changing_the_raw_value() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-SAT-006")["candidate_input"]["normal_raw"] = json!(0.8);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn missing_sample_cannot_be_rewritten_as_a_zero_measurement() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-DROP-012")["candidate_input"]["normal_raw"] = json!(0.0);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn shared_reference_fault_cannot_be_promoted_to_independent_evidence() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-REF-GAIN-023")["evaluator_only"]["expected"]["independence_disposition"] =
            json!("one_physical_root");
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn exact_duplicate_requires_same_sequence_and_same_payload() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-DUP-020")["candidate_input"]["payload_identity"] =
            json!("different-payload");
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn conflicting_duplicate_requires_same_sequence_and_different_payload() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-CONFLICT-021")["candidate_input"]["payload_identity"] =
            json!("payload-021a");
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn hdc_alias_fixture_requires_distinct_raw_traces_with_one_representation() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-ALIAS-030")["candidate_input"]["representation_comparison"]
            ["comparison_normal_raw"] = json!(0.31);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn maximum_array_length_is_enforced_by_the_closed_schema() {
        let (document, mut schema) = tracked_values();
        schema["properties"]["cases"]["maxItems"] = json!(29);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn unknown_nested_fields_are_rejected_by_the_closed_schema() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-NOM-001")["candidate_input"]["unexpected"] = json!(true);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn duplicate_json_object_keys_are_rejected_before_semantic_validation() {
        let raw = r#"{"outer":{"value":1,"value":2}}"#;
        assert!(parse_unique(raw, "duplicate-key fixture").is_err());
    }

    #[test]
    fn clock_offset_cannot_be_mistaken_for_readout_latency() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-CLOCK-017")["candidate_input"]["time_ms"] = json!(1249);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn self_test_pass_does_not_clear_scale_drift() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-SELFTEST-028")["evaluator_only"]["expected"]["qc_flags"] =
            json!(["self_test_insufficient"]);
        assert!(validate_document(&document, &schema).is_err());
    }

    #[test]
    fn lossy_hdc_projection_does_not_replace_the_raw_observation() {
        let (mut document, schema) = tracked_values();
        case_mut(&mut document, "TAC-HDC-029")["candidate_input"]["normal_raw"] = Value::Null;
        assert!(validate_document(&document, &schema).is_err());
    }
}
