use chrono::DateTime;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
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
        "s" | "sec" | "min" | "h" | "day" => Some("time"),
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
        "USD/kW" | "USD/kWh" | "EUR/kW" | "EUR/kWh" => Some("currency_per_capacity"),
        "TEU" => Some("cargo_capacity"),
        "TEU-nautical_mile" => Some("teu_transport_work"),
        "tonne-nautical_mile" => Some("mass_transport_work"),
        "kg/s" | "tonne/day" => Some("mass_flow"),
        "L" | "litre" | "m3/h" => Some("volume_or_volume_flow"),
        "%" => Some("dimensionless"),
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
                    if map
                        .get("unknown_reason")
                        .and_then(Value::as_str)
                        .is_none_or(|s| s.trim().is_empty())
                    {
                        f.push(
                            pointer_child(path, "unknown_reason"),
                            "INVALID",
                            "an unknown quantity must retain an explicit reason",
                        );
                    } else {
                        f.push(
                            path,
                            "INCOMPLETE",
                            "quantity is explicitly unknown; no numeric default is inferred",
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
                        .is_none_or(|v| v <= 0.0)
                    {
                        f.push(
                            pointer_child(path, "value"),
                            "INVALID",
                            "sample rate must be finite and strictly positive",
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

fn check_lineage(case: &Value, f: &mut Findings) {
    let source_ids: BTreeSet<String> = case
        .pointer("/evidence/sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|v| v.get("source_id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut provenance_refs = BTreeSet::new();
    collect_string_refs(case, "source_id", &mut provenance_refs);
    for source in provenance_refs.difference(&source_ids) {
        f.push(
            "/evidence/sources",
            "INCOMPLETE",
            format!("provenance source '{source}' has no corresponding evidence source record"),
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
    collect_string_refs(case, "model_id", &mut model_refs);
    for model in model_refs.difference(&model_ids) {
        f.push(
            "/evidence/model_identities",
            "INCOMPLETE",
            format!("model reference '{model}' has no model identity/revision/applicability record"),
        );
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
                pointer_child(&path, "channel_id"),
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
                    &path,
                    "INVALID",
                    "sample window must have start strictly earlier than end",
                ),
                _ => f.push(
                    &path,
                    "INVALID",
                    "sample window timestamps must be valid RFC3339 date-times",
                ),
            }
        }

        match channel.get("calibration_status").and_then(Value::as_str) {
            Some("in_tolerance") => {}
            Some("out_of_tolerance") => f.push(
                pointer_child(&path, "calibration_status"),
                "INVALID",
                "out-of-tolerance calibration invalidates this measurement channel",
            ),
            Some("stale") | Some("unknown") | Some("not_applicable") | None => f.push(
                pointer_child(&path, "calibration_status"),
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
        "/service/cargo_class",
        "/service/functional_unit",
        "/service/service_contract_id",
        "/service/annual_delivered_work",
        "/service/arrival_policy",
        "/service/required_arrival_window_hours",
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
        "/architecture/powertrain/shaft_boundary_id",
        "/architecture/powertrain/electrical_boundary_id",
        "/economics/currency",
        "/economics/base_year",
        "/economics/lifecycle_years",
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

    #[test]
    fn reviewed_units_have_explicit_dimensions() {
        assert_eq!(unit_dimension("kW"), Some("power"));
        assert_eq!(unit_dimension("nautical_mile"), Some("length"));
        assert_eq!(unit_dimension("gCO2e/MJ"), Some("emissions_intensity"));
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
