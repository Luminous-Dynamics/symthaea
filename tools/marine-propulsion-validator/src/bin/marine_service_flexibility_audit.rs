use chrono::{DateTime, Duration, FixedOffset};
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
struct PortCallResult {
    case_id: String,
    port_call_id: String,
    window_slack_available_seconds: i64,
    window_slack_consumed_seconds: i64,
    window_slack_remaining_seconds: i64,
    early_arrival_margin_seconds: i64,
    planned_arrival_within_window: bool,
    minimum_turnaround_met: bool,
    departure_cutoff_met: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    auditor: &'static str,
    auditor_version: &'static str,
    study_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    cases_checked: usize,
    port_calls_checked: usize,
    port_call_results: Vec<PortCallResult>,
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
        self.diagnostics.push(Diagnostic { path: path.into(), code: code.to_owned(), message: message.into() });
    }

    fn disposition(&self) -> &'static str {
        if self.invalid { "INVALID" }
        else if self.blocked { "BLOCKED" }
        else if self.incomparable { "INCOMPARABLE" }
        else if self.incomplete { "INCOMPLETE" }
        else { "SCHEDULE_CONTRACT_VALID" }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid JSON in {path}: {e}"))
}

fn parse_time(value: Option<&str>) -> Option<DateTime<FixedOffset>> {
    value.and_then(|s| DateTime::parse_from_rfc3339(s).ok())
}

fn validate_schema(schema: &Value, instance: &Value, f: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        f.push("/$schema", "INVALID", "only JSON Schema Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        f.push("/", "INVALID", "schema does not validate against Draft 2020-12 meta-schema");
        return;
    }
    match jsonschema::draft202012::options().should_validate_formats(true).build(schema) {
        Ok(v) => for error in v.iter_errors(instance) {
            f.push(error.instance_path().to_string(), "INVALID", format!("schema violation: {error}"));
        },
        Err(e) => f.push("/", "INVALID", format!("cannot compile schema: {e}")),
    }
}

fn check_registry(case: &Value, service_calls: &BTreeMap<String, Value>, case_index: usize, f: &mut Findings, out: &mut Vec<PortCallResult>) {
    let case_id = case.get("case_id").and_then(Value::as_str).unwrap_or("<missing>").to_owned();
    let calls = case.get("port_calls").and_then(Value::as_array);
    let Some(calls) = calls else {
        f.push(format!("/cases/{case_index}/port_calls"), "INVALID", "port_calls must be an array");
        return;
    };
    let mut seen = BTreeSet::new();
    for (call_index, call) in calls.iter().enumerate() {
        let path = format!("/cases/{case_index}/port_calls/{call_index}");
        let Some(id) = call.get("port_call_id").and_then(Value::as_str) else {
            f.push(format!("{path}/port_call_id"), "INVALID", "port_call_id is missing");
            continue;
        };
        if !seen.insert(id.to_owned()) {
            f.push(format!("{path}/port_call_id"), "INVALID", format!("duplicate port call '{id}'"));
            continue;
        }
        let Some(contract) = service_calls.get(id) else {
            f.push(format!("{path}/port_call_id"), "INVALID", format!("port call '{id}' is not in the service contract"));
            continue;
        };
        let earliest = parse_time(contract.get("earliest_arrival").and_then(Value::as_str));
        let target = parse_time(contract.get("target_arrival").and_then(Value::as_str));
        let latest = parse_time(contract.get("latest_arrival").and_then(Value::as_str));
        let planned_arrival = parse_time(call.get("planned_arrival").and_then(Value::as_str));
        let planned_departure = parse_time(call.get("planned_departure").and_then(Value::as_str));
        let latest_departure = parse_time(contract.get("latest_departure").and_then(Value::as_str));
        let min_turnaround = contract.get("minimum_turnaround_seconds").and_then(Value::as_i64);
        let (Some(earliest), Some(target), Some(latest), Some(arrival), Some(departure), Some(latest_dep), Some(min_turnaround)) =
            (earliest, target, latest, planned_arrival, planned_departure, latest_departure, min_turnaround) else {
                f.push(&path, "INVALID", "all service and planned timestamps must be valid RFC3339 and turnaround must be an integer");
                continue;
        };

        if !(earliest <= target && target <= latest) {
            f.push(&path, "INVALID", "contract window must satisfy earliest_arrival <= target_arrival <= latest_arrival");
            continue;
        }
        if departure < arrival {
            f.push(&path, "INVALID", "planned departure cannot precede planned arrival");
            continue;
        }

        let arrival_ok = arrival >= earliest && arrival <= latest;
        let turnaround_seconds = departure.signed_duration_since(arrival).num_seconds();
        let turnaround_ok = turnaround_seconds >= min_turnaround;
        let departure_ok = departure <= latest_dep;
        if !arrival_ok {
            f.push(format!("{path}/planned_arrival"), "BLOCKED", "planned arrival is outside the hard earliest/latest arrival window");
        }
        if !turnaround_ok {
            f.push(format!("{path}/planned_departure"), "BLOCKED", "planned departure violates minimum turnaround");
        }
        if !departure_ok {
            f.push(format!("{path}/planned_departure"), "BLOCKED", "planned departure exceeds the latest departure cutoff");
        }

        let slack_available = latest.signed_duration_since(target).num_seconds();
        let late_shift = arrival.signed_duration_since(target).num_seconds();
        let consumed = late_shift.max(0).min(slack_available.max(0));
        let remaining = if arrival <= latest { latest.signed_duration_since(arrival).num_seconds().max(0) } else { 0 };
        let early_margin = target.signed_duration_since(arrival).num_seconds().max(0);

        out.push(PortCallResult {
            case_id: case_id.clone(),
            port_call_id: id.to_owned(),
            window_slack_available_seconds: slack_available,
            window_slack_consumed_seconds: consumed,
            window_slack_remaining_seconds: remaining,
            early_arrival_margin_seconds: early_margin,
            planned_arrival_within_window: arrival_ok,
            minimum_turnaround_met: turnaround_ok,
            departure_cutoff_met: departure_ok,
        });
    }

    for id in service_calls.keys() {
        if !seen.contains(id) {
            f.push(format!("/cases/{case_index}/port_calls"), "INVALID", format!("required port call '{id}' is missing"));
        }
    }
}

fn check_semantics(instance: &Value, f: &mut Findings) -> Vec<PortCallResult> {
    let mut results = Vec::new();
    let service = instance.get("service").unwrap_or(&Value::Null);
    let service_fp = service.get("service_fingerprint").and_then(Value::as_str);
    let service_calls = service.get("port_calls").and_then(Value::as_array);
    let Some(service_calls) = service_calls else {
        f.push("/service/port_calls", "INVALID", "service port-call contract is missing");
        return results;
    };

    let mut service_by_id = BTreeMap::new();
    for (i, call) in service_calls.iter().enumerate() {
        let Some(id) = call.get("port_call_id").and_then(Value::as_str) else {
            f.push(format!("/service/port_calls/{i}/port_call_id"), "INVALID", "port_call_id missing");
            continue;
        };
        if service_by_id.insert(id.to_owned(), call.clone()).is_some() {
            f.push(format!("/service/port_calls/{i}/port_call_id"), "INVALID", format!("duplicate service port call '{id}'"));
        }
    }

    let cases = instance.get("cases").and_then(Value::as_array);
    let Some(cases) = cases else { return results };
    let mut ids = BTreeSet::new();
    let mut baseline_count = 0;
    let mut candidate_count = 0;
    for (i, case) in cases.iter().enumerate() {
        let case_id = case.get("case_id").and_then(Value::as_str).unwrap_or("<missing>");
        if !ids.insert(case_id.to_owned()) {
            f.push(format!("/cases/{i}/case_id"), "INVALID", format!("duplicate case id '{case_id}'"));
        }
        match case.get("role").and_then(Value::as_str) {
            Some("baseline") => baseline_count += 1,
            Some("candidate") => candidate_count += 1,
            _ => {}
        }
        if case.get("service_fingerprint").and_then(Value::as_str) != service_fp {
            f.push(format!("/cases/{i}/service_fingerprint"), "INCOMPARABLE", "case service fingerprint differs from the common service contract");
        }
        if instance.get("data_class").and_then(Value::as_str) == Some("synthetic") {
            f.push(format!("/cases/{i}/energy_evidence"), "INCOMPLETE", "synthetic case has no qualifying energy outcome; schedule arithmetic only");
        }
        if case.pointer("/energy_evidence/status").and_then(Value::as_str) == Some("not_supplied") {
            f.push(format!("/cases/{i}/energy_evidence"), "INCOMPLETE", "energy evidence is not supplied; no fuel or storage benefit can be claimed");
        }
        check_registry(case, &service_by_id, i, f, &mut results);
    }
    if baseline_count != 1 {
        f.push("/cases", "INVALID", format!("exactly one baseline required, found {baseline_count}"));
    }
    if candidate_count == 0 {
        f.push("/cases", "INVALID", "at least one candidate is required");
    }
    results
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 { return Err("usage: marine-service-flexibility-audit <schema.json> <study.json>".to_owned()); }
    let schema = read_json(&args[1])?;
    let instance = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &instance, &mut findings);
    let results = check_semantics(&instance, &mut findings);
    let report = Report {
        auditor: "marine-service-flexibility-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        study_id: instance.get("study_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Schedule-window arithmetic and contract checks only. Temporal slack is not energy. This report does not calculate fuel, validate physical models, authorize schedule changes or establish safety/port compliance.",
        cases_checked: instance.get("cases").and_then(Value::as_array).map_or(0, Vec::len),
        port_calls_checked: results.len(),
        port_call_results: results,
        diagnostics: findings.diagnostics,
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(if report.disposition == "SCHEDULE_CONTRACT_VALID" {0} else {2})
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => { eprintln!("{error}"); std::process::exit(64); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-service-flexibility-v0.1.0.schema.json");
    const FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-service-flexibility-v0.1.0.json");

    #[test]
    fn committed_schema_and_fixture_are_valid_and_stay_incomplete() {
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        assert!(jsonschema::draft202012::options().should_validate_formats(true).build(&schema).unwrap().is_valid(&fixture));
        let mut f = Findings::default();
        let results = check_semantics(&fixture, &mut f);
        assert_eq!(results.len(), 6);
        assert_eq!(f.disposition(), "INCOMPLETE");
    }

    #[test]
    fn target_late_arrival_uses_only_declared_window_slack() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let mut f = Findings::default();
        let results = check_semantics(&fixture, &mut f);
        let schedule = results.iter().find(|x| x.case_id == "SYNTHETIC-SCHEDULE-FLEX" && x.port_call_id == "PORT-A").unwrap();
        assert_eq!(schedule.window_slack_available_seconds, 10_800);
        assert_eq!(schedule.window_slack_consumed_seconds, 7_200);
        assert_eq!(schedule.window_slack_remaining_seconds, 3_600);
    }

    #[test]
    fn outside_latest_arrival_blocks_instead_of_becoming_negative_fuel_credit() {
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["cases"][1]["port_calls"][0]["planned_arrival"] = serde_json::json!("2026-12-10T12:30:00Z");
        let mut f = Findings::default();
        let results = check_semantics(&fixture, &mut f);
        assert_eq!(f.disposition(), "BLOCKED");
        assert!(!results.iter().find(|x| x.case_id == "SYNTHETIC-SCHEDULE-FLEX" && x.port_call_id == "PORT-A").unwrap().planned_arrival_within_window);
    }

    #[test]
    fn duplicate_port_call_is_invalid() {
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let first = fixture["cases"][0]["port_calls"][0].clone();
        fixture["cases"][0]["port_calls"].as_array_mut().unwrap().push(first);
        let mut f = Findings::default();
        check_semantics(&fixture, &mut f);
        assert!(f.invalid);
    }

    #[test]
    fn minimum_turnaround_violation_blocks() {
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["cases"][0]["port_calls"][0]["planned_departure"] = serde_json::json!("2026-12-10T10:00:00Z");
        let mut f = Findings::default();
        check_semantics(&fixture, &mut f);
        assert!(f.blocked);
    }

    #[test]
    fn different_service_fingerprint_is_incomparable() {
        let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["cases"][1]["service_fingerprint"] = serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        let mut f = Findings::default();
        check_semantics(&fixture, &mut f);
        assert!(f.incomparable);
    }
}
