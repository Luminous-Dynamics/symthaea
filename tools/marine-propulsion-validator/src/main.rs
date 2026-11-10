use chrono::DateTime;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Serialize)]
struct Diagnostic {
    path: String,
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct Report {
    validator: &'static str,
    validator_version: &'static str,
    disposition: &'static str,
    claim_ceiling: &'static str,
    case_id: Option<String>,
    paired_case_id: Option<String>,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    blocked: bool,
    incomparable: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
            "BLOCKED" => self.blocked = true,
            "INCOMPARABLE" => self.incomparable = true,
            "INCOMPLETE" => self.incomplete = true,
            _ => {}
        }
        self.diagnostics.push(Diagnostic {
            path: path.into(),
            code: code.to_owned(),
            message: message.into(),
        });
    }

    fn disposition(&self) -> &'static str {
        if self.invalid {
            "INVALID"
        } else if self.blocked {
            "BLOCKED"
        } else if self.incomparable {
            "INCOMPARABLE"
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "VALID_FOR_DECLARED_COMPARISON"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON in {path}: {e}"))
}

fn pointer_child(parent: &str, key: &str) -> String {
    let escaped = key.replace('~', "~0").replace('/', "~1");
    format!("{parent}/{escaped}")
}

fn check_schema_refs(schema: &Value, path: &str, f: &mut Findings) {
    match schema {
        Value::Object(map) => {
            for (key, value) in map {
                if key == "$dynamicRef" || key == "$recursiveRef" {
                    f.push(
                        pointer_child(path, key),
                        "INVALID",
                        "dynamic or recursive reference semantics are not admitted by this validator",
                    );
                } else if key == "$ref" {
                    match value.as_str() {
                        Some(reference) if reference.starts_with('#') => {}
                        _ => f.push(
                            pointer_child(path, key),
                            "INVALID",
                            "only in-document fragment references are allowed; network and filesystem schema retrieval are disabled",
                        ),
                    }
                }
                check_schema_refs(value, &pointer_child(path, key), f);
            }
        }
        Value::Array(items) => {
            for (index, value) in items.iter().enumerate() {
                check_schema_refs(value, &format!("{path}/{index}"), f);
            }
        }
        _ => {}
    }
}

fn unit_dimension(unit: &str) -> Option<&'static str> {
    match unit {
        "1" | "dimensionless" | "count" => Some("dimensionless"),
        "m" | "km" | "nautical_mile" | "nm" => Some("length"),
        "m2" | "m^2" => Some("area"),
        "m3" | "m^3" => Some("volume"),
        "kg" | "tonne" | "t" => Some("mass"),
        "s" | "sec" | "ms" | "us" | "min" | "h" | "day" => Some("time"),
        "Hz" | "1/s" => Some("frequency"),
        "m/s" | "knot" | "kn" => Some("velocity"),
        "m/s2" | "m/s^2" => Some("acceleration"),
        "rad" | "deg" | "degree" => Some("angle"),
        "rpm" => Some("rotational_frequency"),
        "N" | "kN" => Some("force"),
        "Nm" | "N.m" | "kNm" => Some("moment"),
        "Pa" | "kPa" | "MPa" | "bar" => Some("pressure"),
        "W" | "kW" | "MW" => Some("power"),
        "J" | "kJ" | "MJ" | "GJ" | "kWh" | "MWh" => Some("energy"),
        "J/kg" | "kJ/kg" | "MJ/kg" => Some("specific_energy"),
        "gCO2e/MJ" | "kgCO2e/MJ" | "gCO2e/kWh" => Some("emissions_intensity"),
        "USD" | "EUR" | "ZAR" => Some("currency"),
        "USD/year" | "EUR/year" | "ZAR/year" => Some("currency_per_time"),
        "USD/kW" | "EUR/kW" => Some("currency_per_power"),
        "USD/kWh" | "EUR/kWh" => Some("currency_per_energy"),
        "ppm" => Some("dimensionless_ratio"),
        "TEU" => Some("cargo_capacity"),
        "TEU-nautical_mile" => Some("teu_transport_work"),
        "tonne-nautical_mile" => Some("mass_transport_work"),
        "kg/s" | "tonne/day" => Some("mass_flow"),
        "L" | "litre" => Some("volume"),
        "m3/h" => Some("volume_flow"),
        "%" | "ppm" => Some("dimensionless"),
        _ => None,
    }
}

fn check_uncertainty(value: &Value, path: &str, f: &mut Findings) {
    let Some(kind) = value.get("kind").and_then(Value::as_str) else {
        return;
    };
    match kind {
        "interval" => {
            let lower = value.get("lower").and_then(Value::as_f64);
            let upper = value.get("upper").and_then(Value::as_f64);
            match (lower, upper) {
                (Some(lo), Some(hi)) if lo <= hi => {}
                (Some(_), Some(_)) => f.push(
                    path,
                    "INVALID",
                    "uncertainty interval lower bound exceeds upper bound",
                ),
                _ => f.push(
                    path,
                    "INVALID",
                    "interval uncertainty requires numeric lower and upper bounds",
                ),
            }
        }
        "standard" => {
            if value
                .get("standard_uncertainty")
                .and_then(Value::as_f64)
                .is_some_and(|v| v < 0.0)
            {
                f.push(
                    pointer_child(path, "standard_uncertainty"),
                    "INVALID",
                    "standard uncertainty cannot be negative",
                );
            }
        }
        "unknown" => {
            if value
                .get("reason")
                .and_then(Value::as_str)
                .is_none_or(|s| s.trim().is_empty())
            {
                f.push(
                    pointer_child(path, "reason"),
                    "INVALID",
                    "unknown uncertainty must retain an explicit reason",
                );
            } else {
                f.push(
                    path,
                    "INCOMPLETE",
                    "uncertainty is explicitly unknown; it remains unknown and is not treated as zero",
                );
            }
        }
        _ => {}
    }
}

fn walk_semantics(value: &Value, path: &str, f: &mut Findings) {
    match value {
        Value::Object(map) => {
            if map.contains_key("unit") && map.contains_key("value") {
                if let Some(unit) = map.get("unit").and_then(Value::as_str) {
                    match unit_dimension(unit) {
                        None => f.push(
                            pointer_child(path, "unit"),
                            "INVALID",
                            format!("unit '{unit}' is outside the validator's reviewed vocabulary"),
                        ),
                        Some(dimension) => {
                            if path.ends_with("/sample_rate_hz") && dimension != "frequency" {
                                f.push(
                                    pointer_child(path, "unit"),
                                    "INVALID",
                                    "sample_rate_hz must use a frequency unit (Hz or 1/s)",
                                );
                            }
                            if path.ends_with("/time_alignment_uncertainty")
                                && dimension != "time"
                            {
                                f.push(
                                    pointer_child(path, "unit"),
                                    "INVALID",
                                    "time alignment uncertainty must use a time unit",
                                );
                            }
                        }
                    }
                }
                if map.get("value").is_some_and(Value::is_null) {
                    let not_applicable = map
                        .get("not_applicable_reason")
                        .and_then(Value::as_str)
                        .is_some_and(|reason| !reason.trim().is_empty());
                    let unknown = map
                        .get("unknown_reason")
                        .and_then(Value::as_str)
                        .is_some_and(|reason| !reason.trim().is_empty());
                    if not_applicable {
                        if path == "/service/annual_delivered_work" || path.ends_with("/distance") {
                            f.push(
                                path.clone(),
                                "INCOMPLETE",
                                "a required transport-work or route-distance quantity cannot be not applicable",
                            );
                        }
                        // Explicit non-applicability is a declared state, not missing data.
                    } else if unknown {
                        f.push(
                            path,
                            "INCOMPLETE",
                            "quantity is explicitly unknown; no numeric default is inferred",
                        );
                    } else {
                        f.push(
                            path,
                            "INVALID",
                            "a null quantity must carry unknown_reason or not_applicable_reason",
                        );
                    }
                }
                if let Some(uncertainty) = map.get("uncertainty") {
                    check_uncertainty(uncertainty, &pointer_child(path, "uncertainty"), f);
                }
                if path.ends_with("/sample_rate_hz") {
                    if map
                        .get("value")
                        .and_then(Value::as_f64)
                        .is_some_and(|v| v <= 0.0)
                    {
                        f.push(
                            pointer_child(path, "value"),
                            "INVALID",
                            "sample rate must be strictly positive",
                        );
                    }
                    if map.get("not_applicable_reason").is_some() {
                        f.push(
                            pointer_child(path, "not_applicable_reason"),
                            "INVALID",
                            "a required measurement channel sample rate cannot be not applicable",
                        );
                    }
                }
                if path.ends_with("/time_alignment_uncertainty")
                    && map
                        .get("value")
                        .and_then(Value::as_f64)
                        .is_some_and(|v| v < 0.0)
                {
                    f.push(
                        pointer_child(path, "value"),
                        "INVALID",
                        "time alignment uncertainty cannot be negative",
                    );
                }
            }
            for (key, child) in map {
                walk_semantics(child, &pointer_child(path, key), f);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                walk_semantics(child, &format!("{path}/{index}"), f);
            }
        }
        _ => {}
    }
}

fn unique_ids_at(root: &Value, pointer: &str, field: &str, f: &mut Findings) {
    let Some(items) = root.pointer(pointer).and_then(Value::as_array) else {
        return;
    };
    let mut seen = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        if let Some(id) = item.get(field).and_then(Value::as_str) {
            if !seen.insert(id.to_owned()) {
                f.push(
                    format!("{pointer}/{index}/{field}"),
                    "INVALID",
                    format!("duplicate {field} '{id}' in this identity scope"),
                );
            }
        }
    }
}

fn collect_string_refs(value: &Value, key: &str, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            if let Some(candidate) = map.get(key).and_then(Value::as_str) {
                out.insert(candidate.to_owned());
            }
            for child in map.values() {
                collect_string_refs(child, key, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_string_refs(child, key, out);
            }
        }
        _ => {}
    }
}

fn collect_provenance_source_refs(value: &Value, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            if let Some(source_id) = map
                .get("provenance")
                .and_then(|provenance| provenance.get("source_id"))
                .and_then(Value::as_str)
            {
                out.insert(source_id.to_owned());
            }
            for child in map.values() {
                collect_provenance_source_refs(child, out);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_provenance_source_refs(child, out);
            }
        }
        _ => {}
    }
}

fn check_lineage(case: &Value, f: &mut Findings) {
    let source_ids: BTreeSet<String> = case
        .pointer("/evidence/sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|v| v.get("source_id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut provenance_refs = BTreeSet::new();
    collect_provenance_source_refs(case, &mut provenance_refs);
    let mut evidence_refs = BTreeSet::new();
    collect_string_refs(case, "evidence_source_id", &mut evidence_refs);
    provenance_refs.extend(evidence_refs);
    for source in provenance_refs.difference(&source_ids) {
        f.push(
            "/evidence/sources",
            "INCOMPLETE",
            format!("provenance/evidence source '{source}' has no corresponding evidence source record"),
        );
    }

    let model_ids: BTreeSet<String> = case
        .pointer("/evidence/model_identities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|v| v.get("model_id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut model_refs = BTreeSet::new();
    for key in [
        "model_id",
        "resistance_model_id",
        "propulsion_model_id",
        "control_model_id",
        "degradation_model_id",
    ] {
        collect_string_refs(case, key, &mut model_refs);
    }
    for model in model_refs.difference(&model_ids) {
        f.push(
            "/evidence/model_identities",
            "INCOMPLETE",
            format!("model reference '{model}' has no model identity/revision/applicability record"),
        );
    }
}

const SERVICE_FINGERPRINT_PROFILE: &str =
    "marine-service-payload-v1+symthaea-canonical-json-sha256-v1";

fn write_canonical_json(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => {
            output.push_str(
                &serde_json::to_string(value).expect("serializing a JSON string cannot fail"),
            );
        }
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(value, output);
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key).expect("serializing a JSON key cannot fail"),
                );
                output.push(':');
                write_canonical_json(value, output);
            }
            output.push('}');
        }
    }
}

fn canonical_json_sha256(value: &Value) -> String {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical);
    let digest = Sha256::digest(canonical.as_bytes());
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn required_field<'a>(value: &'a Value, field: &str, path: &str) -> Result<&'a Value, String> {
    value.get(field).ok_or_else(|| format!("missing required service fingerprint field {path}/{field}"))
}

fn projected_quantity(value: &Value) -> Value {
    let mut projected = Map::new();
    for field in ["value", "unit", "unknown_reason", "not_applicable_reason"] {
        if let Some(member) = value.get(field) {
            projected.insert(field.to_owned(), member.clone());
        }
    }
    Value::Object(projected)
}

fn insert_optional_quantity(target: &mut Map<String, Value>, source: &Value, field: &str) {
    if let Some(quantity) = source.get(field) {
        target.insert(field.to_owned(), projected_quantity(quantity));
    }
}

fn service_fingerprint_payload(case: &Value) -> Result<Value, String> {
    let comparison = required_field(case, "comparison", "")?;
    let profile = comparison
        .get("service_fingerprint_profile")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing service_fingerprint_profile".to_owned())?;
    if profile != SERVICE_FINGERPRINT_PROFILE {
        return Err(format!("unsupported service fingerprint profile '{profile}'"));
    }

    let service = required_field(case, "service", "")?;
    let route = required_field(case, "route", "")?;
    let vessel = required_field(case, "vessel", "")?;
    let mut payload = Map::new();
    payload.insert(
        "fingerprint_profile".to_owned(),
        Value::String(SERVICE_FINGERPRINT_PROFILE.to_owned()),
    );

    for field in ["cargo_class", "functional_unit", "service_contract_id", "arrival_policy"] {
        payload.insert(field.to_owned(), required_field(service, field, "/service")?.clone());
    }
    if let Some(other_unit) = service.get("declared_other_unit") {
        payload.insert("declared_other_unit".to_owned(), other_unit.clone());
    }
    for field in [
        "annual_delivered_work",
        "annual_delivered_cargo_mass",
        "annual_delivered_teu",
        "required_arrival_window_hours",
        "service_availability_target",
    ] {
        insert_optional_quantity(&mut payload, service, field);
    }

    payload.insert(
        "loading_condition".to_owned(),
        required_field(vessel, "loading_condition", "/vessel")?.clone(),
    );
    payload.insert("route_id".to_owned(), required_field(route, "route_id", "/route")?.clone());
    payload.insert(
        "route_revision".to_owned(),
        required_field(route, "route_revision", "/route")?.clone(),
    );

    let segments = route
        .get("segments")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing route segments for service fingerprint".to_owned())?;
    let mut projected_segments = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        let path = format!("/route/segments/{index}");
        let mut projected = Map::new();
        for field in ["segment_id", "speed_policy"] {
            projected.insert(
                field.to_owned(),
                required_field(segment, field, &path)?.clone(),
            );
        }
        let distance = required_field(segment, "distance", &path)?;
        projected.insert("distance".to_owned(), projected_quantity(distance));
        for field in ["target_speed", "arrival_time_window"] {
            insert_optional_quantity(&mut projected, segment, field);
        }
        if let Some(port) = segment.get("port_or_waypoint_id") {
            projected.insert("port_or_waypoint_id".to_owned(), port.clone());
        }
        projected_segments.push(Value::Object(projected));
    }
    payload.insert("segments".to_owned(), Value::Array(projected_segments));
    Ok(Value::Object(payload))
}

fn compute_service_fingerprint(case: &Value) -> Result<String, String> {
    let payload = service_fingerprint_payload(case)?;
    Ok(canonical_json_sha256(&payload))
}

fn check_service_fingerprint(case: &Value, f: &mut Findings) {
    let expected = case
        .pointer("/comparison/service_fingerprint")
        .and_then(Value::as_str);
    let profile = case
        .pointer("/comparison/service_fingerprint_profile")
        .and_then(Value::as_str);
    match profile {
        None => f.push(
            "/comparison/service_fingerprint_profile",
            "INCOMPLETE",
            "legacy case has no versioned canonical service fingerprint profile; digest equality alone is not independently recomputed",
        ),
        Some(profile) if profile != SERVICE_FINGERPRINT_PROFILE => f.push(
            "/comparison/service_fingerprint_profile",
            "INVALID",
            format!("unsupported service fingerprint profile '{profile}'"),
        ),
        Some(_) => match compute_service_fingerprint(case) {
            Ok(actual) if Some(actual.as_str()) == expected => {}
            Ok(_) => f.push(
                "/comparison/service_fingerprint",
                "INVALID",
                "service fingerprint does not match the canonical service/route/loading payload",
            ),
            Err(error) => f.push(
                "/comparison/service_fingerprint",
                "INVALID",
                error,
            ),
        },
    }
}

fn time_to_seconds(unit: &str) -> Option<f64> {
    match unit {
        "s" | "sec" => Some(1.0),
        "ms" => Some(1.0e-3),
        "us" => Some(1.0e-6),
        "min" => Some(60.0),
        "h" => Some(3600.0),
        "day" => Some(86400.0),
        _ => None,
    }
}

fn frequency_to_hz(unit: &str) -> Option<f64> {
    match unit {
        "Hz" | "1/s" => Some(1.0),
        _ => None,
    }
}

fn check_trial(case: &Value, f: &mut Findings) {
    let Some(trial) = case.get("trial_evidence") else {
        return;
    };
    if trial.is_null() {
        return;
    }

    let expected_fingerprint = case
        .pointer("/comparison/service_fingerprint")
        .and_then(Value::as_str);
    let trial_fingerprint = trial.get("service_fingerprint").and_then(Value::as_str);
    if expected_fingerprint != trial_fingerprint {
        f.push(
            "/trial_evidence/service_fingerprint",
            "INCOMPARABLE",
            "trial service fingerprint does not equal the case comparison fingerprint",
        );
    }

    let calibration_records = trial
        .get("calibration_records")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let timebases = trial
        .get("timebases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let calibration_by_id: BTreeMap<String, Value> = calibration_records
        .iter()
        .filter_map(|record| {
            record
                .get("calibration_record_id")
                .and_then(Value::as_str)
                .map(|id| (id.to_owned(), record.clone()))
        })
        .collect();
    let timebase_by_id: BTreeMap<String, Value> = timebases
        .iter()
        .filter_map(|record| {
            record
                .get("timebase_id")
                .and_then(Value::as_str)
                .map(|id| (id.to_owned(), record.clone()))
        })
        .collect();
    let minimum_observed_fraction = trial
        .pointer("/sample_completeness_policy/minimum_observed_fraction")
        .and_then(Value::as_f64);
    let schema_version = case.get("schema_version").and_then(Value::as_str).unwrap_or("");
    let is_v3 = schema_version.starts_with("0.3.") || schema_version.starts_with("0.4.");
    let is_v4 = schema_version.starts_with("0.4.");
    let source_ids: BTreeSet<String> = case
        .pointer("/evidence/sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| value.get("source_id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let rate_count_tolerance = trial
        .pointer("/sample_completeness_policy/rate_count_tolerance_samples")
        .and_then(Value::as_f64);

    if minimum_observed_fraction.is_some_and(|v| v <= 0.0 || v > 1.0) {
        f.push(
            "/trial_evidence/sample_completeness_policy/minimum_observed_fraction",
            "INVALID",
            "minimum observed sample fraction must be greater than zero and no greater than one",
        );
    }

    if let Some(excluded) = trial.get("excluded_intervals").and_then(Value::as_array) {
        for (index, interval) in excluded.iter().enumerate() {
            let path = format!("/trial_evidence/excluded_intervals/{index}");
            let start = interval.get("start").and_then(Value::as_str);
            let end = interval.get("end").and_then(Value::as_str);
            if let (Some(start), Some(end)) = (start, end) {
                match (
                    DateTime::parse_from_rfc3339(start),
                    DateTime::parse_from_rfc3339(end),
                ) {
                    (Ok(start), Ok(end)) if start < end => {}
                    (Ok(_), Ok(_)) => f.push(
                        path.clone(),
                        "INVALID",
                        "excluded interval must have start strictly earlier than end",
                    ),
                    _ => f.push(
                        path.clone(),
                        "INVALID",
                        "excluded interval timestamps must be valid RFC3339 date-times",
                    ),
                }
            }
        }
    }

    let channels = trial.get("channels").and_then(Value::as_array);
    let Some(channels) = channels else {
        return;
    };
    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ids = BTreeSet::new();

    for (index, channel) in channels.iter().enumerate() {
        let path = format!("/trial_evidence/channels/{index}");
        let Some(id) = channel.get("channel_id").and_then(Value::as_str) else {
            continue;
        };
        if !ids.insert(id.to_owned()) {
            f.push(
                pointer_child(path.clone(), "channel_id"),
                "INVALID",
                format!("duplicate measurement channel id '{id}'"),
            );
        }

        let start = channel.get("sample_window_start").and_then(Value::as_str);
        let end = channel.get("sample_window_end").and_then(Value::as_str);
        if let (Some(start), Some(end)) = (start, end) {
            match (
                DateTime::parse_from_rfc3339(start),
                DateTime::parse_from_rfc3339(end),
            ) {
                (Ok(start), Ok(end)) if start < end => {}
                (Ok(_), Ok(_)) => f.push(
                    path.clone(),
                    "INVALID",
                    "sample window must have start strictly earlier than end",
                ),
                _ => f.push(
                    path.clone(),
                    "INVALID",
                    "sample window timestamps must be valid RFC3339 date-times",
                ),
            }
        }

        let expected_samples = channel.get("expected_sample_count").and_then(Value::as_u64);
        let observed_samples = channel.get("observed_sample_count").and_then(Value::as_u64);
        let missing_samples = channel.get("missing_sample_count").and_then(Value::as_u64);
        if let (Some(expected), Some(observed), Some(missing)) =
            (expected_samples, observed_samples, missing_samples)
        {
            if expected == 0 || observed.saturating_add(missing) != expected {
                f.push(
                    path.clone(),
                    "INVALID",
                    "sample counts must satisfy expected > 0 and observed + missing = expected",
                );
            } else if let Some(minimum) = minimum_observed_fraction {
                let coverage = observed as f64 / expected as f64;
                if coverage < minimum {
                    f.push(
                        pointer_child(path.clone(), "observed_sample_count"),
                        "INCOMPLETE",
                        format!(
                            "observed sample coverage {coverage:.6} is below the predeclared minimum {minimum:.6}"
                        ),
                    );
                }
            }
        }

        let window = match (start, end) {
            (Some(start), Some(end)) => match (
                DateTime::parse_from_rfc3339(start),
                DateTime::parse_from_rfc3339(end),
            ) {
                (Ok(start), Ok(end)) if start < end => Some((start, end)),
                _ => None,
            },
            _ => None,
        };

        if is_v3 {
            match channel.get("sample_rate_basis").and_then(Value::as_str) {
                Some("nominal") | Some("measured") => {}
                _ => f.push(
                    pointer_child(path.clone(), "sample_rate_basis"),
                    "INVALID",
                    "v0.3 channel must declare whether its sampling rate is nominal or measured",
                ),
            }
            if let (Some((window_start, window_end)), Some(expected), Some(tolerance)) =
                (window.as_ref(), expected_samples, rate_count_tolerance)
            {
                let sample_rate = channel
                    .pointer("/sample_rate_hz/value")
                    .and_then(Value::as_f64);
                let sample_rate_unit = channel
                    .pointer("/sample_rate_hz/unit")
                    .and_then(Value::as_str);
                match (sample_rate, sample_rate_unit.and_then(frequency_to_hz)) {
                    (Some(rate), Some(scale)) if rate > 0.0 => {
                        let duration_ns = window_end.clone().signed_duration_since(window_start.clone()).num_nanoseconds();
                        match duration_ns {
                            Some(duration_ns) if duration_ns > 0 => {
                                let expected_from_rate = duration_ns as f64 / 1_000_000_000.0 * rate * scale;
                                let difference = (expected as f64 - expected_from_rate).abs();
                                if difference > tolerance {
                                    f.push(
                                        pointer_child(path.clone(), "expected_sample_count"),
                                        "INVALID",
                                        format!(
                                            "declared expected sample count {expected} differs from rate/window estimate {expected_from_rate:.6} by more than the policy tolerance {tolerance:.6} samples"
                                        ),
                                    );
                                }
                            }
                            _ => f.push(
                                pointer_child(path.clone(), "sample_window_end"),
                                "INVALID",
                                "sample window duration cannot be represented as a positive nanosecond interval",
                            ),
                        }
                    }
                    _ => f.push(
                        pointer_child(path.clone(), "sample_rate_hz"),
                        "INVALID",
                        "v0.3 sample-count validation requires a known positive sample rate in Hz or 1/s",
                    ),
                }
            }
        }

        if let Some(calibration_id) = channel.get("calibration_record_id").and_then(Value::as_str) {
            match calibration_by_id.get(calibration_id) {
                None if !calibration_records.is_empty() => f.push(
                    pointer_child(path.clone(), "calibration_record_id"),
                    "INVALID",
                    format!("calibration record '{calibration_id}' does not resolve in trial registry"),
                ),
                None => f.push(
                    pointer_child(path.clone(), "calibration_record_id"),
                    "INCOMPLETE",
                    "calibration record is not resolvable because this trial has no calibration registry",
                ),
                Some(record) => {
                    let record_path = format!(
                        "/trial_evidence/calibration_records/{}",
                        calibration_records.iter().position(|r| r.get("calibration_record_id").and_then(Value::as_str)==Some(calibration_id)).unwrap_or(0)
                    );
                    if is_v4 {
                        let reference_epoch = record
                            .get("reference_epoch")
                            .and_then(Value::as_str)
                            .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                        match reference_epoch.as_ref() {
                            Some(epoch) => {
                                let valid_from = record.get("valid_from").and_then(Value::as_str)
                                    .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                                let valid_until = record.get("valid_until").and_then(Value::as_str)
                                    .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                                match (valid_from, valid_until) {
                                    (Some(from), Some(until)) if epoch < &from || epoch > &until => f.push(
                                        pointer_child(record_path.clone(), "reference_epoch"),
                                        "INVALID",
                                        "timebase reference_epoch must lie within the timebase validity interval",
                                    ),
                                    (Some(_), Some(_)) => {}
                                    _ => f.push(
                                        pointer_child(record_path.clone(), "reference_epoch"),
                                        "INCOMPLETE",
                                        "timebase reference epoch cannot be checked because validity dates are unresolved",
                                    ),
                                }
                            }
                            None => f.push(
                                pointer_child(record_path.clone(), "reference_epoch"),
                                "INVALID",
                                "v0.4 timebase must declare a valid RFC3339 reference_epoch",
                            ),
                        }
                        if record.get("drift_semantics").and_then(Value::as_str)
                            != Some("residual_rate_uncertainty_after_correction")
                        {
                            f.push(
                                pointer_child(record_path.clone(), "drift_semantics"),
                                "INVALID",
                                "v0.4 drift uncertainty must be declared as residual rate uncertainty after correction",
                            );
                        }
                        let drift_unit = record.pointer("/drift_uncertainty_ppm/unit").and_then(Value::as_str);
                        if drift_unit != Some("ppm") {
                            f.push(
                                pointer_child(record_path.clone(), "drift_uncertainty_ppm"),
                                "INVALID",
                                "residual timebase rate uncertainty must use the reviewed ppm unit",
                            );
                        }
                        if record.pointer("/drift_uncertainty_ppm/value").and_then(Value::as_f64)
                            .is_some_and(|value| value < 0.0)
                        {
                            f.push(
                                pointer_child(record_path.clone(), "drift_uncertainty_ppm"),
                                "INVALID",
                                "residual drift-rate uncertainty cannot be negative",
                            );
                        }
                        if record.pointer("/drift_uncertainty_ppm/value").and_then(Value::as_f64).is_none() {
                            f.push(
                                pointer_child(record_path.clone(), "drift_uncertainty_ppm"),
                                "INCOMPLETE",
                                "residual drift-rate uncertainty is unknown; it cannot be treated as zero",
                            );
                        }
                    }

                    if is_v3 {
                        match record.get("evidence_source_id").and_then(Value::as_str) {
                            Some(source_id) if source_ids.contains(source_id) => {}
                            Some(source_id) => f.push(
                                pointer_child(record_path.clone(), "evidence_source_id"),
                                "INVALID",
                                format!("calibration evidence source '{source_id}' does not resolve in /evidence/sources"),
                            ),
                            None => f.push(
                                pointer_child(record_path.clone(), "evidence_source_id"),
                                "INVALID",
                                "v0.3 calibration record must identify its evidence source",
                            ),
                        }
                        match record.get("rights_status").and_then(Value::as_str) {
                            Some("public") | Some("permissioned") | Some("restricted") => {}
                            Some("unknown") | Some("not_applicable") | None => f.push(
                                pointer_child(record_path.clone(), "rights_status"),
                                "INCOMPLETE",
                                "calibration certificate rights/access status is unknown or not applicable",
                            ),
                            _ => {}
                        }
                        if record.get("certificate_sha256").and_then(Value::as_str).is_none() {
                            f.push(
                                pointer_child(record_path.clone(), "certificate_sha256"),
                                "INCOMPLETE",
                                "calibration certificate digest is absent",
                            );
                        }
                    }
                    if record.get("physical_quantity") != channel.get("physical_quantity")
                        || record.get("unit") != channel.get("unit")
                    {
                        f.push(
                            pointer_child(path.clone(), "unit"),
                            "INVALID",
                            "calibration quantity and unit must exactly match the measurement channel",
                        );
                    }
                    for identity_field in ["id", "revision"] {
                        if record.pointer(&format!("/instrument_identity/{identity_field}"))
                            != channel.pointer(&format!("/instrument_identity/{identity_field}"))
                        {
                            f.push(
                                pointer_child(path.clone(), "instrument_identity"),
                                "INVALID",
                                format!("calibration instrument {identity_field} does not match channel instrument"),
                            );
                            break;
                        }
                    }
                    let range_min = record.get("range_min").and_then(Value::as_f64);
                    let range_max = record.get("range_max").and_then(Value::as_f64);
                    if matches!((range_min, range_max), (Some(lo), Some(hi)) if lo > hi) {
                        f.push(
                            record_path.clone(),
                            "INVALID",
                            "calibration range_min must not exceed range_max",
                        );
                    }
                    let status = record.get("disposition").and_then(Value::as_str);
                    if status != channel.get("calibration_status").and_then(Value::as_str) {
                        f.push(
                            pointer_child(path.clone(), "calibration_status"),
                            "INVALID",
                            "channel calibration_status must match the referenced calibration record disposition",
                        );
                    }
                    match status {
                        Some("in_tolerance") => {}
                        Some("out_of_tolerance") => f.push(
                            pointer_child(path.clone(), "calibration_record_id"),
                            "INVALID",
                            "referenced calibration is out of tolerance",
                        ),
                        Some("stale") | Some("unknown") | None => f.push(
                            pointer_child(path.clone(), "calibration_record_id"),
                            "INCOMPLETE",
                            "referenced calibration is stale or unknown",
                        ),
                        _ => {}
                    }
                    if let Some((sample_start, sample_end)) = window.clone() {
                        let valid_from = record.get("valid_from").and_then(Value::as_str)
                            .and_then(|v| DateTime::parse_from_rfc3339(v).ok());
                        let valid_until = record.get("valid_until").and_then(Value::as_str)
                            .and_then(|v| DateTime::parse_from_rfc3339(v).ok());
                        let calibration_date = record.get("calibration_date").and_then(Value::as_str)
                            .and_then(|v| DateTime::parse_from_rfc3339(v).ok());
                        if let (Some(calibrated), Some(from)) =
                            (calibration_date.as_ref(), valid_from.as_ref())
                        {
                            if calibrated > from {
                                f.push(
                                    record_path.clone(),
                                    "INVALID",
                                    "calibration_date must not be later than valid_from",
                                );
                            }
                        }
                        match (valid_from, valid_until) {
                            (Some(from), Some(until)) if from <= sample_start && until >= sample_end => {}
                            (Some(from), Some(until)) if from >= until => f.push(
                                record_path.clone(),
                                "INVALID",
                                "calibration validity interval must have valid_from earlier than valid_until",
                            ),
                            _ => f.push(
                                record_path.clone(),
                                "INCOMPLETE",
                                "calibration validity does not cover the complete measurement sample window",
                            ),
                        }
                    }
                }
            }
        }

        if let Some(timebase_id) = channel.get("timebase_id").and_then(Value::as_str) {
            match timebase_by_id.get(timebase_id) {
                None if !timebases.is_empty() => f.push(
                    pointer_child(path.clone(), "timebase_id"),
                    "INVALID",
                    format!("timebase '{timebase_id}' does not resolve in trial registry"),
                ),
                None => f.push(
                    pointer_child(path.clone(), "timebase_id"),
                    "INCOMPLETE",
                    "timebase is not resolvable because this trial has no timebase registry",
                ),
                Some(record) => {
                    let record_path = format!(
                        "/trial_evidence/timebases/{}",
                        timebases.iter().position(|r| r.get("timebase_id").and_then(Value::as_str)==Some(timebase_id)).unwrap_or(0)
                    );
                    if is_v3 {
                        match record.get("evidence_source_id").and_then(Value::as_str) {
                            Some(source_id) if source_ids.contains(source_id) => {}
                            Some(source_id) => f.push(
                                pointer_child(record_path.clone(), "evidence_source_id"),
                                "INVALID",
                                format!("timebase evidence source '{source_id}' does not resolve in /evidence/sources"),
                            ),
                            None => f.push(
                                pointer_child(record_path.clone(), "evidence_source_id"),
                                "INVALID",
                                "v0.3 timebase record must identify its evidence source",
                            ),
                        }
                        match record.get("rights_status").and_then(Value::as_str) {
                            Some("public") | Some("permissioned") | Some("restricted") => {}
                            Some("unknown") | Some("not_applicable") | None => f.push(
                                pointer_child(record_path.clone(), "rights_status"),
                                "INCOMPLETE",
                                "timebase evidence rights/access status is unknown or not applicable",
                            ),
                            _ => {}
                        }
                    }
                    if let Some((sample_start, sample_end)) = window.clone() {
                        let valid_from = record.get("valid_from").and_then(Value::as_str)
                            .and_then(|v| DateTime::parse_from_rfc3339(v).ok());
                        let valid_until = record.get("valid_until").and_then(Value::as_str)
                            .and_then(|v| DateTime::parse_from_rfc3339(v).ok());
                        match (valid_from, valid_until) {
                            (Some(from), Some(until)) if from <= sample_start && until >= sample_end => {}
                            (Some(from), Some(until)) if from >= until => f.push(
                                record_path.clone(),
                                "INVALID",
                                "timebase validity interval must have valid_from earlier than valid_until",
                            ),
                            _ => f.push(
                                record_path.clone(),
                                "INCOMPLETE",
                                "timebase validity does not cover the complete measurement sample window",
                            ),
                        }
                    }

                    let channel_uncertainty = channel
                        .pointer("/time_alignment_uncertainty/value")
                        .and_then(Value::as_f64);
                    let channel_unit = channel
                        .pointer("/time_alignment_uncertainty/unit")
                        .and_then(Value::as_str);
                    let clock_uncertainty = record
                        .pointer("/synchronization_uncertainty/value")
                        .and_then(Value::as_f64);
                    let clock_unit = record
                        .pointer("/synchronization_uncertainty/unit")
                        .and_then(Value::as_str);
                    if let (Some(channel_value), Some(channel_unit), Some(clock_value), Some(clock_unit)) =
                        (channel_uncertainty, channel_unit, clock_uncertainty, clock_unit)
                    {
                        let channel_seconds = time_to_seconds(channel_unit).map(|scale| channel_value * scale);
                        let base_clock_seconds = time_to_seconds(clock_unit).map(|scale| clock_value * scale);
                        let required_clock_seconds = if is_v4 {
                            match (
                                base_clock_seconds,
                                record.pointer("/drift_uncertainty_ppm/value").and_then(Value::as_f64),
                                record.get("reference_epoch").and_then(Value::as_str)
                                    .and_then(|value| DateTime::parse_from_rfc3339(value).ok()),
                                window.as_ref(),
                            ) {
                                (Some(base), Some(drift_ppm), Some(epoch), Some((sample_start, sample_end))) => {
                                    let start_elapsed = sample_start.clone()
                                        .signed_duration_since(epoch.clone())
                                        .num_nanoseconds()
                                        .map(|value| (value as f64).abs() / 1_000_000_000.0);
                                    let end_elapsed = sample_end.clone()
                                        .signed_duration_since(epoch)
                                        .num_nanoseconds()
                                        .map(|value| (value as f64).abs() / 1_000_000_000.0);
                                    match (start_elapsed, end_elapsed) {
                                        (Some(start), Some(end)) => {
                                            let elapsed_max = start.max(end);
                                            Some(base + drift_ppm * 1.0e-6 * elapsed_max)
                                        }
                                        _ => {
                                            f.push(
                                                pointer_child(path.clone(), "time_alignment_uncertainty"),
                                                "INCOMPLETE",
                                                "cannot calculate accumulated drift uncertainty for this sample window",
                                            );
                                            None
                                        }
                                    }
                                }
                                _ => {
                                    f.push(
                                        pointer_child(path.clone(), "time_alignment_uncertainty"),
                                        "INCOMPLETE",
                                        "v0.4 timing check requires known base uncertainty, residual drift uncertainty, reference epoch and sample window",
                                    );
                                    None
                                }
                            }
                        } else {
                            base_clock_seconds
                        };
                        match (channel_seconds, required_clock_seconds) {
                            (Some(channel_bound), Some(required_bound))
                                if channel_bound + f64::EPSILON >= required_bound => {}
                            (Some(_), Some(_)) => f.push(
                                pointer_child(path.clone(), "time_alignment_uncertainty"),
                                "INVALID",
                                "channel alignment uncertainty understates the timebase reference uncertainty plus accumulated residual drift bound",
                            ),
                            _ => f.push(
                                pointer_child(path.clone(), "time_alignment_uncertainty"),
                                "INVALID",
                                "time alignment uncertainty units must be convertible using the reviewed time vocabulary",
                            ),
                        }
                    } else {
                        f.push(
                            pointer_child(path.clone(), "time_alignment_uncertainty"),
                            "INCOMPLETE",
                            "channel and timebase synchronization uncertainty must both be known",
                        );
                    }
                }
            }
        }

        match channel.get("calibration_status").and_then(Value::as_str) {
            Some("in_tolerance") => {}
            Some("out_of_tolerance") => f.push(
                pointer_child(path.clone(), "calibration_status"),
                "INVALID",
                "out-of-tolerance calibration invalidates this measurement channel",
            ),
            Some("stale") | Some("unknown") | Some("not_applicable") | None => f.push(
                pointer_child(path.clone(), "calibration_status"),
                "INCOMPLETE",
                "calibration is not evidenced as in tolerance",
            ),
            _ => {}
        }

        let parents: Vec<String> = channel
            .get("derived_from_channel_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        graph.insert(id.to_owned(), parents);
    }

    for (id, parents) in &graph {
        for parent in parents {
            if !graph.contains_key(parent) {
                f.push(
                    "/trial_evidence/channels",
                    "INVALID",
                    format!("derived channel '{id}' references missing channel '{parent}'"),
                );
            }
        }
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in graph.keys() {
        if has_cycle(id, &graph, &mut visiting, &mut visited) {
            f.push(
                "/trial_evidence/channels",
                "INVALID",
                "derived-channel dependency graph contains a cycle",
            );
            break;
        }
    }
}

fn has_cycle(
    id: &str,
    graph: &BTreeMap<String, Vec<String>>,
    visiting: &mut BTreeSet<String>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if visited.contains(id) {
        return false;
    }
    if !visiting.insert(id.to_owned()) {
        return true;
    }
    if let Some(parents) = graph.get(id) {
        for parent in parents {
            if graph.contains_key(parent)
                && has_cycle(parent, graph, visiting, visited)
            {
                return true;
            }
        }
    }
    visiting.remove(id);
    visited.insert(id.to_owned());
    false
}

fn check_unique_scopes(case: &Value, f: &mut Findings) {
    unique_ids_at(case, "/hard_gates", "gate_id", f);
    unique_ids_at(case, "/route/segments", "segment_id", f);
    unique_ids_at(case, "/architecture/powertrain/stages", "stage_id", f);
    unique_ids_at(case, "/architecture/energy_sources", "source_id", f);
    unique_ids_at(case, "/architecture/wind_devices", "device_id", f);
    unique_ids_at(case, "/evidence/sources", "source_id", f);
    unique_ids_at(case, "/evidence/model_identities", "model_id", f);
    unique_ids_at(case, "/candidate_metrics", "metric_id", f);
    unique_ids_at(case, "/trial_evidence/calibration_records", "calibration_record_id", f);
    unique_ids_at(case, "/trial_evidence/timebases", "timebase_id", f);
}

fn check_gates(case: &Value, f: &mut Findings) {
    let Some(gates) = case.get("hard_gates").and_then(Value::as_array) else {
        return;
    };
    for (index, gate) in gates.iter().enumerate() {
        let path = format!("/hard_gates/{index}/status");
        match gate.get("status").and_then(Value::as_str) {
            Some("fail") | Some("blocked") => f.push(
                path,
                "BLOCKED",
                "a required hard gate is failed or blocked; no performance ranking is permitted",
            ),
            Some("unknown") => f.push(
                path,
                "INCOMPLETE",
                "hard-gate status is unknown; unknown is not a pass",
            ),
            _ => {}
        }
    }
}

fn compare_quantity(left: &Value, right: &Value, pointer: &str, label: &str, f: &mut Findings) {
    let left_quantity = left.pointer(pointer);
    let right_quantity = right.pointer(pointer);
    let compact = |value: Option<&Value>| {
        value.map(|q| (q.get("value").cloned(), q.get("unit").cloned()))
    };
    if compact(left_quantity) != compact(right_quantity) {
        f.push(
            pointer,
            "INCOMPARABLE",
            format!("paired cases differ in {label} value or unit"),
        );
    }
}

fn compare_pointer(
    left: &Value,
    right: &Value,
    pointer: &str,
    label: &str,
    f: &mut Findings,
) {
    if left.pointer(pointer) != right.pointer(pointer) {
        f.push(
            pointer,
            "INCOMPARABLE",
            format!("paired cases differ in {label}; no reviewed comparability profile is declared"),
        );
    }
}

fn check_pair(left: &Value, right: &Value, f: &mut Findings) {
    let left_id = left.get("case_id").and_then(Value::as_str).unwrap_or("<unknown>");
    let right_id = right.get("case_id").and_then(Value::as_str).unwrap_or("<unknown>");
    let left_role = left.pointer("/comparison/role").and_then(Value::as_str);
    let right_role = right.pointer("/comparison/role").and_then(Value::as_str);
    let roles_match = matches!(
        (left_role, right_role),
        (Some("baseline"), Some("candidate")) | (Some("candidate"), Some("baseline"))
    );
    if !roles_match {
        f.push(
            "/comparison/role",
            "INCOMPARABLE",
            "a comparison pair must contain exactly one baseline and one candidate; reference_only cases cannot be ranked",
        );
    }

    if left.pointer("/comparison/comparison_group_id")
        != right.pointer("/comparison/comparison_group_id")
    {
        f.push(
            "/comparison/comparison_group_id",
            "INCOMPARABLE",
            "paired cases must belong to the same comparison group",
        );
    }

    if left.pointer("/comparison/service_fingerprint")
        != right.pointer("/comparison/service_fingerprint")
    {
        f.push(
            "/comparison/service_fingerprint",
            "INCOMPARABLE",
            "baseline and candidate service fingerprints differ",
        );
    }

    let left_pair = left.pointer("/comparison/paired_case_id").and_then(Value::as_str);
    let right_pair = right.pointer("/comparison/paired_case_id").and_then(Value::as_str);
    if left_pair != Some(right_id) || right_pair != Some(left_id) {
        f.push(
            "/comparison/paired_case_id",
            "INCOMPARABLE",
            "paired_case_id links must be reciprocal and resolve to the supplied case ids",
        );
    }

    for (case, role, id) in [(left, left_role, left_id), (right, right_role, right_id)] {
        if case
            .pointer("/comparison/matching_deviations")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
        {
            f.push(
                "/comparison/matching_deviations",
                "INCOMPARABLE",
                format!("case '{id}' declares unmatched service deviations; no reviewed profile is supplied"),
            );
        }
        if role == Some("reference_only") {
            f.push(
                "/comparison/role",
                "INCOMPARABLE",
                format!("case '{id}' is reference_only and cannot be ranked"),
            );
        }
    }

    for pointer in [
        "/service/annual_delivered_work",
        "/service/annual_delivered_cargo_mass",
        "/service/annual_delivered_teu",
        "/service/service_availability_target",
        "/service/required_arrival_window_hours",
    ] {
        compare_quantity(left, right, pointer, pointer, f);
    }

    for pointer in [
        "/service/cargo_class",
        "/service/functional_unit",
        "/service/service_contract_id",
        "/service/arrival_policy",
        "/vessel/loading_condition",
        "/route/route_id",
        "/route/route_revision",
        "/route/segments",
        "/environment/profile_id",
        "/environment/profile_revision",
        "/environment/time_basis",
        "/environment/wind_series_sha256",
        "/environment/wave_series_sha256",
        "/environment/current_series_sha256",
        "/environment/water_depth_series_sha256",
        "/architecture/powertrain/shaft_boundary_id",
        "/architecture/powertrain/electrical_boundary_id",
        "/economics/currency",
        "/economics/base_year",
        "/economics/lifecycle_years",
        "/economics/discount_rate",
        "/evidence/data_class",
    ] {
        compare_pointer(left, right, pointer, pointer, f);
    }
}

fn validate_instance(schema: &Value, instance: &Value, label: &str, f: &mut Findings) {
    let Some(schema_uri) = schema.get("$schema").and_then(Value::as_str) else {
        f.push("/$schema", "INVALID", "schema must declare its dialect");
        return;
    };
    if schema_uri != DRAFT_2020_12 {
        f.push(
            "/$schema",
            "INVALID",
            "only JSON Schema Draft 2020-12 is supported",
        );
        return;
    }

    if !jsonschema::draft202012::meta::is_valid(schema) {
        f.push(
            "/",
            "INVALID",
            "schema document does not validate against the Draft 2020-12 meta-schema",
        );
        return;
    }

    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            f.push("/", "INVALID", format!("cannot compile schema: {error}"));
            return;
        }
    };

    for error in validator.iter_errors(instance) {
        f.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("{label}: {error}"),
        );
    }
}

fn validate_semantics(case: &Value, f: &mut Findings, pair_supplied: bool) {
    if case.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        f.push(
            "/record_status",
            "INCOMPLETE",
            "synthetic records are software fixtures and cannot establish physical performance",
        );
    }

    walk_semantics(case, "", f);
    check_service_fingerprint(case, f);
    check_unique_scopes(case, f);
    check_lineage(case, f);
    check_trial(case, f);
    check_gates(case, f);

    if case
        .pointer("/comparison/paired_case_id")
        .and_then(Value::as_str)
        .is_some()
        && !pair_supplied
    {
        f.push(
            "/comparison/paired_case_id",
            "INCOMPLETE",
            "paired_case_id is declared but no paired case was supplied to this invocation",
        );
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        return Err(
            "usage: marine-propulsion-validator <schema.json> <case.json> [paired-case.json]"
                .to_owned(),
        );
    }

    let schema = read_json(&args[1])?;
    let case = read_json(&args[2])?;
    let paired = if args.len() == 4 {
        Some(read_json(&args[3])?)
    } else {
        None
    };

    let mut findings = Findings::default();
    check_schema_refs(&schema, "", &mut findings);

    let schema_major = schema
        .get("$id")
        .and_then(Value::as_str)
        .and_then(|id| id.rsplit(':').next())
        .and_then(|version| version.split('.').next());
    let case_major = case
        .get("schema_version")
        .and_then(Value::as_str)
        .and_then(|version| version.split('.').next());
    if schema_major.is_none() || case_major.is_none() || schema_major != case_major {
        findings.push(
            "/schema_version",
            "INVALID",
            "case and schema major versions are absent or incompatible",
        );
    }

    validate_instance(&schema, &case, "case", &mut findings);
    if let Some(paired_case) = paired.as_ref() {
        validate_instance(&schema, paired_case, "paired case", &mut findings);
        validate_semantics(&case, &mut findings, true);
        validate_semantics(paired_case, &mut findings, true);
        check_pair(&case, paired_case, &mut findings);
    } else {
        validate_semantics(&case, &mut findings, false);
    }

    let report = Report {
        validator: "marine-propulsion-validator",
        validator_version: env!("CARGO_PKG_VERSION"),
        disposition: findings.disposition(),
        claim_ceiling: "Contract and declared-comparability checks only. This report does not validate physics, performance, safety, seaworthiness, class/flag compliance, sensor authenticity, or operational authority.",
        case_id: case.get("case_id").and_then(Value::as_str).map(str::to_owned),
        paired_case_id: paired
            .as_ref()
            .and_then(|v| v.get("case_id"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        diagnostics: findings.diagnostics,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );

    Ok(if report.disposition == "VALID_FOR_DECLARED_COMPARISON" {
        0
    } else {
        2
    })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MARINE_SCHEMA: &str =
        include_str!("../../../docs/research/marine/marine-propulsion-case.schema.json");
    const SYNTHETIC_CASE: &str =
        include_str!("../../../docs/research/marine/fixtures/synthetic-waps-case-0001.json");

    const MARINE_SCHEMA_V2: &str =
        include_str!("../../../docs/research/marine/marine-propulsion-case-v0.2.0.schema.json");
    const SYNTHETIC_CASE_V2: &str =
        include_str!("../../../docs/research/marine/fixtures/synthetic-waps-case-v0.2.0.json");
    const MARINE_SCHEMA_V3: &str =
        include_str!("../../../docs/research/marine/marine-propulsion-case-v0.3.0.schema.json");
    const SYNTHETIC_CASE_V3: &str =
        include_str!("../../../docs/research/marine/fixtures/synthetic-waps-case-v0.3.0.json");
    const MARINE_SCHEMA_V4: &str =
        include_str!("../../../docs/research/marine/marine-propulsion-case-v0.4.0.schema.json");
    const SYNTHETIC_CASE_V4: &str =
        include_str!("../../../docs/research/marine/fixtures/synthetic-waps-case-v0.4.0.json");
    const SERVICE_FINGERPRINT_VECTOR_V1: &str =
        include_str!("../../../docs/research/marine/fixtures/service-fingerprint-v1-vector-0001.json");

    fn synthetic_matched_pair() -> (Value, Value) {
        let fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        let mut baseline = fixture.clone();
        let mut candidate = fixture;
        baseline["case_id"] = serde_json::json!("SYNTHETIC-BASELINE-0001");
        candidate["case_id"] = serde_json::json!("SYNTHETIC-CANDIDATE-0001");
        baseline["comparison"]["role"] = serde_json::json!("baseline");
        candidate["comparison"]["role"] = serde_json::json!("candidate");
        baseline["comparison"]["paired_case_id"] =
            serde_json::json!("SYNTHETIC-CANDIDATE-0001");
        candidate["comparison"]["paired_case_id"] =
            serde_json::json!("SYNTHETIC-BASELINE-0001");
        (baseline, candidate)
    }

    #[test]
    fn published_service_fingerprint_vector_matches_canonical_bytes_and_digest() {
        let vector: Value = serde_json::from_str(SERVICE_FINGERPRINT_VECTOR_V1)
            .expect("service fingerprint vector JSON");
        let payload = vector.get("payload").expect("payload");
        let mut canonical = String::new();
        write_canonical_json(payload, &mut canonical);
        assert_eq!(
            canonical,
            vector.get("canonical_json").and_then(Value::as_str).expect("canonical_json")
        );
        assert_eq!(
            canonical_json_sha256(payload),
            vector.get("sha256").and_then(Value::as_str).expect("sha256")
        );
        assert_eq!(
            vector.get("profile").and_then(Value::as_str),
            Some(SERVICE_FINGERPRINT_PROFILE)
        );
    }

    #[test]
    fn service_fingerprint_matches_known_canonical_vector() {
        let fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        assert_eq!(
            compute_service_fingerprint(&fixture).expect("canonical fingerprint"),
            "e3ec23f5308213b6d7ea3743e6cf483623436bec5384e29d3044d525d736fca9"
        );
    }

    #[test]
    fn service_fingerprint_changes_when_declared_service_changes() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        let original = compute_service_fingerprint(&fixture).expect("original fingerprint");
        fixture["route"]["segments"][0]["target_speed"]["value"] =
            serde_json::json!(19);
        let changed = compute_service_fingerprint(&fixture).expect("changed fingerprint");
        assert_ne!(original, changed);
    }

    #[test]
    fn canonical_json_fingerprint_ignores_object_key_order() {
        let left: Value = serde_json::from_str(r#"{"b":2,"a":{"d":4,"c":3}}"#).unwrap();
        let right: Value = serde_json::from_str(r#"{"a":{"c":3,"d":4},"b":2}"#).unwrap();
        assert_eq!(canonical_json_sha256(&left), canonical_json_sha256(&right));
    }

    #[test]
    fn required_transport_work_cannot_be_not_applicable() {
        let mut findings = Findings::default();
        let quantity = serde_json::json!({
            "value": null,
            "unit": "TEU-nautical_mile",
            "not_applicable_reason": "None",
            "provenance": {
                "class":"synthetic",
                "source_id":"SYNTHETIC-WAPS-FIXTURE-ONLY",
                "method":"test",
                "limitations":["test-only"],
                "rights_status":"not_applicable"
            }
        });
        walk_semantics(&quantity, "/service/annual_delivered_work", &mut findings);
        assert!(findings.incomplete);
    }

    #[test]
    fn explicit_not_applicable_quantity_is_not_misclassified_as_unknown() {
        let mut findings = Findings::default();
        let quantity = serde_json::json!({
            "value": null,
            "unit": "tonne",
            "not_applicable_reason": "This contract uses TEU transport work.",
            "provenance": {
                "class":"synthetic",
                "source_id":"SYNTHETIC-WAPS-FIXTURE-ONLY",
                "method":"test",
                "limitations":["test-only"],
                "rights_status":"not_applicable"
            }
        });
        walk_semantics(&quantity, "/service/annual_delivered_cargo_mass", &mut findings);
        assert!(!findings.invalid);
        assert!(!findings.incomplete);
    }

    #[test]
    fn service_fingerprint_distinguishes_not_applicable_from_unknown_and_zero() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");

        fixture["service"]["annual_delivered_cargo_mass"]["value"] = serde_json::json!(0);
        let numeric_zero = compute_service_fingerprint(&fixture).expect("zero fingerprint");

        fixture["service"]["annual_delivered_cargo_mass"] = serde_json::json!({
            "value": null,
            "unit": "tonne",
            "not_applicable_reason": "The contract functional unit is TEU-nautical-mile.",
            "provenance": {
                "class":"synthetic",
                "source_id":"SYNTHETIC-WAPS-FIXTURE-ONLY",
                "method":"test mutation",
                "limitations":["Test-only state distinction."],
                "rights_status":"not_applicable"
            }
        });
        let not_applicable = compute_service_fingerprint(&fixture).expect("not-applicable fingerprint");
        assert_ne!(numeric_zero, not_applicable);

        fixture["service"]["annual_delivered_cargo_mass"] = serde_json::json!({
            "value": null,
            "unit": "tonne",
            "unknown_reason": "Cargo mass was not measured.",
            "provenance": {
                "class":"synthetic",
                "source_id":"SYNTHETIC-WAPS-FIXTURE-ONLY",
                "method":"test mutation",
                "limitations":["Test-only state distinction."],
                "rights_status":"not_applicable"
            }
        });
        let unknown = compute_service_fingerprint(&fixture).expect("unknown fingerprint");
        assert_ne!(not_applicable, unknown);

        fixture["service"].as_object_mut().expect("service object")
            .remove("annual_delivered_cargo_mass");
        let omitted = compute_service_fingerprint(&fixture).expect("omitted fingerprint");
        assert_ne!(unknown, omitted);
        assert_ne!(not_applicable, omitted);
    }

    #[test]
    fn v0_2_schema_and_fixture_resolve_calibration_and_timebase_contracts() {
        let schema: Value = serde_json::from_str(MARINE_SCHEMA_V2).expect("v0.2 schema JSON");
        let fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("v0.2 schema compiles");
        assert!(validator.is_valid(&fixture));
        let mut findings = Findings::default();
        check_unique_scopes(&fixture, &mut findings);
        check_trial(&fixture, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn v0_3_schema_and_fixture_validate_with_provenance_and_sampling_policy() {
        let schema: Value = serde_json::from_str(MARINE_SCHEMA_V3).expect("v0.3 schema JSON");
        let fixture: Value = serde_json::from_str(SYNTHETIC_CASE_V3).expect("v0.3 fixture JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("v0.3 schema compiles");
        assert!(validator.is_valid(&fixture));
        let mut findings = Findings::default();
        check_unique_scopes(&fixture, &mut findings);
        check_trial(&fixture, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
        assert!(findings.incomplete); // synthetic evidence and rights remain non-qualifying
    }

    #[test]
    fn v0_4_schema_and_fixture_validate_reference_epoch_and_drift_contract() {
        let schema: Value = serde_json::from_str(MARINE_SCHEMA_V4).expect("v0.4 schema JSON");
        let fixture: Value = serde_json::from_str(SYNTHETIC_CASE_V4).expect("v0.4 fixture JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("v0.4 schema compiles");
        assert!(validator.is_valid(&fixture));
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
        assert!(findings.incomplete);
    }

    #[test]
    fn v0_4_longer_elapsed_time_increases_required_alignment_bound() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V4).expect("v0.4 fixture JSON");
        fixture["trial_evidence"]["timebases"][0]["reference_epoch"] =
            serde_json::json!("2026-11-10T11:00:00Z");
        fixture["trial_evidence"]["timebases"][0]["drift_uncertainty_ppm"]["value"] =
            serde_json::json!(100);
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn v0_4_channel_bound_covering_accumulated_drift_is_not_invalid() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V4).expect("v0.4 fixture JSON");
        fixture["trial_evidence"]["timebases"][0]["reference_epoch"] =
            serde_json::json!("2026-11-10T11:00:00Z");
        fixture["trial_evidence"]["timebases"][0]["drift_uncertainty_ppm"]["value"] =
            serde_json::json!(100);
        fixture["trial_evidence"]["channels"][0]["time_alignment_uncertainty"]["value"] =
            serde_json::json!(0.008);
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn v0_4_unknown_drift_uncertainty_is_incomplete_not_zero() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V4).expect("v0.4 fixture JSON");
        fixture["trial_evidence"]["timebases"][0]["drift_uncertainty_ppm"]["value"] =
            Value::Null;
        fixture["trial_evidence"]["timebases"][0]["drift_uncertainty_ppm"]["unknown_reason"] =
            serde_json::json!("Residual clock-rate uncertainty is not measured.");
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn v0_4_reference_epoch_must_be_inside_validity_window() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V4).expect("v0.4 fixture JSON");
        fixture["trial_evidence"]["timebases"][0]["reference_epoch"] =
            serde_json::json!("2025-01-01T00:00:00Z");
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_3_unknown_certificate_rights_are_incomplete() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V3).expect("v0.3 fixture JSON");
        fixture["trial_evidence"]["calibration_records"][0]["rights_status"] =
            serde_json::json!("unknown");
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn v0_3_unresolved_calibration_evidence_source_is_invalid() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V3).expect("v0.3 fixture JSON");
        fixture["trial_evidence"]["calibration_records"][0]["evidence_source_id"] =
            serde_json::json!("MISSING-SOURCE-RECORD");
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_3_declared_sample_count_must_agree_with_rate_and_window() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V3).expect("v0.3 fixture JSON");
        fixture["trial_evidence"]["channels"][0]["sample_rate_hz"]["value"] =
            serde_json::json!(2);
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn compatible_time_units_are_compared_after_explicit_conversion() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V3).expect("v0.3 fixture JSON");
        fixture["trial_evidence"]["timebases"][0]["synchronization_uncertainty"]["value"] =
            serde_json::json!(2);
        fixture["trial_evidence"]["timebases"][0]["synchronization_uncertainty"]["unit"] =
            serde_json::json!("ms");
        // Channel uncertainty remains 0.002 s; the physical uncertainties are equal.
        let mut findings = Findings::default();
        check_trial(&fixture, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn required_channel_sample_rate_cannot_be_not_applicable() {
        let mut findings = Findings::default();
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        fixture["trial_evidence"]["channels"][0]["sample_rate_hz"] = serde_json::json!({
            "value":null,
            "unit":"Hz",
            "not_applicable_reason":"not supplied",
            "provenance":{
                "class":"synthetic",
                "source_id":"SYNTHETIC-WAPS-FIXTURE-ONLY",
                "method":"test",
                "limitations":["test-only"],
                "rights_status":"not_applicable"
            }
        });
        walk_semantics(&fixture, "", &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_2_sample_count_mismatch_is_invalid() {
        let mut findings = Findings::default();
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        fixture["trial_evidence"]["channels"][0]["observed_sample_count"] =
            serde_json::json!(59);
        let schema: Value = serde_json::from_str(MARINE_SCHEMA_V2).expect("v0.2 schema JSON");
        validate_instance(&schema, &fixture, "case", &mut findings);
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_2_coverage_below_policy_is_incomplete_not_a_pass() {
        let mut findings = Findings::default();
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        fixture["trial_evidence"]["channels"][0]["observed_sample_count"] =
            serde_json::json!(57);
        fixture["trial_evidence"]["channels"][0]["missing_sample_count"] =
            serde_json::json!(3);
        check_trial(&fixture, &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn v0_2_calibration_instrument_mismatch_is_invalid() {
        let mut findings = Findings::default();
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        fixture["trial_evidence"]["calibration_records"][0]["instrument_identity"]["revision"] =
            serde_json::json!("wrong-revision");
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_2_channel_cannot_understate_timebase_uncertainty() {
        let mut findings = Findings::default();
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        fixture["trial_evidence"]["channels"][0]["time_alignment_uncertainty"]["value"] =
            serde_json::json!(0.001);
        check_trial(&fixture, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn reciprocal_matched_pair_is_eligible_for_comparability_only() {
        let (baseline, candidate) = synthetic_matched_pair();
        let mut findings = Findings::default();
        validate_semantics(&baseline, &mut findings, true);
        validate_semantics(&candidate, &mut findings, true);
        check_pair(&baseline, &candidate, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
        assert!(!findings.incomparable, "{:?}", findings.diagnostics);
        assert!(findings.incomplete); // synthetic evidence and unknown gates remain non-qualified
    }

    #[test]
    fn route_segment_order_is_part_of_service_identity() {
        let mut fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE_V2).expect("v0.2 fixture JSON");
        let mut second = fixture["route"]["segments"][0].clone();
        second["segment_id"] = serde_json::json!("SEG-002");
        second["distance"]["value"] = serde_json::json!(1200);
        second["port_or_waypoint_id"] = serde_json::json!("SYNTHETIC-PORT-002");
        fixture["route"]["segments"].as_array_mut().expect("segments").push(second);
        let forward = compute_service_fingerprint(&fixture).expect("forward route digest");
        fixture["route"]["segments"].as_array_mut().expect("segments").reverse();
        let reverse = compute_service_fingerprint(&fixture).expect("reverse route digest");
        assert_ne!(forward, reverse);
    }

    #[test]
    fn changed_route_with_recomputed_hash_is_incomparable() {
        let (baseline, mut candidate) = synthetic_matched_pair();
        candidate["route"]["segments"][0]["target_speed"]["value"] =
            serde_json::json!(19);
        let digest = compute_service_fingerprint(&candidate).expect("candidate fingerprint");
        candidate["comparison"]["service_fingerprint"] = serde_json::json!(digest);
        candidate["trial_evidence"]["service_fingerprint"] = serde_json::json!(digest);
        let mut findings = Findings::default();
        validate_semantics(&baseline, &mut findings, true);
        validate_semantics(&candidate, &mut findings, true);
        check_pair(&baseline, &candidate, &mut findings);
        assert!(findings.incomparable, "{:?}", findings.diagnostics);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn stale_fingerprint_after_service_mutation_is_invalid() {
        let (_, mut candidate) = synthetic_matched_pair();
        candidate["service"]["service_contract_id"] =
            serde_json::json!("MUTATED-CONTRACT");
        let mut findings = Findings::default();
        check_service_fingerprint(&candidate, &mut findings);
        assert!(findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn metocean_profile_remains_a_separate_pair_boundary() {
        let (baseline, mut candidate) = synthetic_matched_pair();
        candidate["environment"]["profile_revision"] =
            serde_json::json!("changed-weather-profile");
        // The declared service fingerprint intentionally remains identical.
        assert_eq!(
            compute_service_fingerprint(&baseline).unwrap(),
            compute_service_fingerprint(&candidate).unwrap()
        );
        let mut findings = Findings::default();
        check_pair(&baseline, &candidate, &mut findings);
        assert!(findings.incomparable, "{:?}", findings.diagnostics);
    }

    #[test]
    fn committed_schema_and_fixture_pass_structural_validation() {
        let schema: Value = serde_json::from_str(MARINE_SCHEMA).expect("committed schema JSON");
        let fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE).expect("committed fixture JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("schema compiles");
        assert!(validator.is_valid(&fixture));
    }

    #[test]
    fn synthetic_fixture_is_never_a_performance_qualification() {
        let fixture: Value =
            serde_json::from_str(SYNTHETIC_CASE).expect("committed fixture JSON");
        let mut findings = Findings::default();
        validate_semantics(&fixture, &mut findings, false);
        assert_eq!(findings.disposition(), "INCOMPLETE");
    }

    #[test]
    fn reviewed_units_have_explicit_dimensions() {
        assert_eq!(unit_dimension("kW"), Some("power"));
        assert_eq!(unit_dimension("nautical_mile"), Some("length"));
        assert_eq!(unit_dimension("gCO2e/MJ"), Some("emissions_intensity"));
        assert_eq!(unit_dimension("L"), Some("volume"));
        assert_eq!(unit_dimension("m3/h"), Some("volume_flow"));
        assert_eq!(unit_dimension("USD/kW"), Some("currency_per_power"));
        assert_eq!(unit_dimension("USD/kWh"), Some("currency_per_energy"));
        assert_eq!(unit_dimension("invented-unit"), None);
    }

    #[test]
    fn unknown_units_are_not_inferred_from_labels() {
        let mut findings = Findings::default();
        let value = serde_json::json!({
            "value": 10.0,
            "unit": "ship-efficiency-ish",
            "provenance": {},
            "uncertainty": {"kind":"unknown","reason":"not measured"}
        });
        walk_semantics(&value, "/probe", &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn reversed_uncertainty_interval_is_invalid() {
        let mut findings = Findings::default();
        check_uncertainty(
            &serde_json::json!({
                "kind":"interval",
                "lower": 12.0,
                "upper": 10.0
            }),
            "/probe/uncertainty",
            &mut findings,
        );
        assert!(findings.invalid);
    }

    #[test]
    fn unknown_uncertainty_remains_incomplete_not_zero() {
        let mut findings = Findings::default();
        check_uncertainty(
            &serde_json::json!({"kind":"unknown","reason":"no calibrated data"}),
            "/probe/uncertainty",
            &mut findings,
        );
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn dependency_cycle_is_detected() {
        let graph = BTreeMap::from([
            ("a".to_owned(), vec!["b".to_owned()]),
            ("b".to_owned(), vec!["a".to_owned()]),
        ]);
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        assert!(has_cycle("a", &graph, &mut visiting, &mut visited));
    }

    #[test]
    fn disposition_precedence_never_hides_invalid_data() {
        let mut findings = Findings::default();
        findings.push("/gate", "BLOCKED", "a gate is blocked");
        findings.push("/value", "INVALID", "malformed data");
        assert_eq!(findings.disposition(), "INVALID");
    }
}
