use chrono::{DateTime, NaiveDateTime};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Serialize)]
struct Diagnostic {
    path: String,
    code: String,
    message: String,
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
            "SCENARIO_ACCOUNTING_VALID"
        }
    }
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    scenario_id: Option<String>,
    port_call_id: Option<String>,
    record_status: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    delivery_eligible: bool,
    usable_connection_seconds: u64,
    effective_connection_power_w: u64,
    requested_energy_wh: u64,
    delivered_energy_wh: u64,
    unserved_energy_wh: u64,
    grid_ghg_micro_kg_co2e: Option<u128>,
    displaced_generator_ghg_micro_kg_co2e: Option<u128>,
    net_ghg_delta_micro_kg_co2e: Option<i128>,
    diagnostics: Vec<Diagnostic>,
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn parse_timestamp(value: Option<&str>) -> Option<DateTime<chrono::FixedOffset>> {
    value.and_then(|text| DateTime::parse_from_rfc3339(text).ok())
}

fn validate_schema(schema: &Value, instance: &Value, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push("/$schema", "INVALID", "only JSON Schema Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push("/", "INVALID", "schema does not validate against Draft 2020-12 meta-schema");
        return;
    }
    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            findings.push("/", "INVALID", format!("cannot compile schema: {error}"));
            return;
        }
    };
    for error in validator.iter_errors(instance) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("schema violation: {error}"),
        );
    }
}

fn min_u64(values: &[u64]) -> u64 {
    values.iter().copied().min().unwrap_or(0)
}

fn ghg_for_energy(energy_wh: u64, factor_micro_kg_per_kwh: u64) -> Option<u128> {
    // Wh × (micro-kg/kWh) / 1,000 Wh/kWh. Integer division rounds down.
    (energy_wh as u128)
        .checked_mul(factor_micro_kg_per_kwh as u128)
        .map(|product| product / 1_000)
}

fn check_source_registry(case: &Value, findings: &mut Findings) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    if let Some(sources) = case.get("evidence_sources").and_then(Value::as_array) {
        for (index, source) in sources.iter().enumerate() {
            let path = format!("/evidence_sources/{index}/source_id");
            match source.get("source_id").and_then(Value::as_str) {
                Some(id) if !ids.insert(id.to_owned()) => {
                    findings.push(path, "INVALID", format!("duplicate evidence source '{id}'"));
                }
                Some(_) => {}
                None => findings.push(path, "INVALID", "source_id is missing"),
            }
        }
    }
    ids
}

fn check_factor_source(
    factor: &Value,
    path: &str,
    source_ids: &BTreeSet<String>,
    findings: &mut Findings,
) {
    match factor.get("source_id").and_then(Value::as_str) {
        Some(id) if source_ids.contains(id) => {}
        Some(id) => findings.push(
            format!("{path}/source_id"),
            "INVALID",
            format!("factor source '{id}' does not resolve in /evidence_sources"),
        ),
        None => findings.push(
            format!("{path}/source_id"),
            "INVALID",
            "factor must reference an evidence source",
        ),
    }
}

fn run_audit(schema: &Value, case: &Value) -> AuditReport {
    let mut findings = Findings::default();
    validate_schema(schema, case, &mut findings);

    if case.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "synthetic port-call evidence is a software fixture, not operational OPS qualification",
        );
    }

    let source_ids = check_source_registry(case, &mut findings);
    let berth = case.get("berth_window").unwrap_or(&Value::Null);
    let connection = case.get("connection").unwrap_or(&Value::Null);
    let demand = case.get("demand").unwrap_or(&Value::Null);
    let supply = case.get("port_supply").unwrap_or(&Value::Null);

    let requested_energy_wh = demand
        .get("requested_energy_wh")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let requested_max_power_w = demand
        .get("requested_max_power_w")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ship_max_power_w = connection
        .get("ship_max_input_power_w")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let berth_limit_w = connection
        .get("berth_connection_limit_w")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reserved_power_w = supply
        .get("reserved_power_w")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let available_energy_wh = supply
        .get("available_energy_wh")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let connection_time = berth
        .get("connection_time_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let disconnection_time = berth
        .get("disconnection_time_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let berth_start = parse_timestamp(berth.get("berth_start").and_then(Value::as_str));
    let berth_end = parse_timestamp(berth.get("berth_end").and_then(Value::as_str));
    let arrival = parse_timestamp(berth.get("planned_arrival").and_then(Value::as_str));
    let departure = parse_timestamp(berth.get("planned_departure").and_then(Value::as_str));

    let mut berth_duration_seconds = 0u64;
    match (berth_start, berth_end, arrival, departure) {
        (Some(start), Some(end), Some(arrive), Some(depart)) => {
            if end <= start || depart <= arrive || arrive < start || depart > end {
                findings.push(
                    "/berth_window",
                    "INVALID",
                    "planned arrival/departure must be ordered and contained within the declared berth window",
                );
            } else {
                let seconds = depart.signed_duration_since(arrive).num_seconds();
                if seconds < 0 {
                    findings.push(
                        "/berth_window/planned_departure",
                        "INVALID",
                        "planned departure precedes arrival",
                    );
                } else {
                    berth_duration_seconds = seconds as u64;
                }
            }
        }
        _ => findings.push(
            "/berth_window",
            "INVALID",
            "all berth/arrival/departure timestamps must be valid RFC3339 values",
        ),
    }

    let overhead_seconds = connection_time.saturating_add(disconnection_time);
    if overhead_seconds > berth_duration_seconds {
        findings.push(
            "/berth_window/connection_time_seconds",
            "BLOCKED",
            "connection and disconnection overhead exceed the berth stay; no connection time remains",
        );
    }
    let usable_connection_seconds = berth_duration_seconds.saturating_sub(overhead_seconds);

    let compatibility = connection.get("compatibility").and_then(Value::as_str);
    match compatibility {
        Some("compatible") => {}
        Some("incompatible") => findings.push(
            "/connection/compatibility",
            "BLOCKED",
            "ship and berth shore-power interfaces are incompatible",
        ),
        Some("unknown") | None => findings.push(
            "/connection/compatibility",
            "INCOMPLETE",
            "ship/berth compatibility is not established",
        ),
        _ => findings.push(
            "/connection/compatibility",
            "INVALID",
            "unsupported compatibility disposition",
        ),
    }

    match connection.get("port_availability").and_then(Value::as_str) {
        Some("available") => {}
        Some("unavailable") => findings.push(
            "/connection/port_availability",
            "BLOCKED",
            "shore-power facility is unavailable for this call",
        ),
        Some("unknown") | None => findings.push(
            "/connection/port_availability",
            "INCOMPLETE",
            "shore-power facility availability is unknown",
        ),
        _ => findings.push(
            "/connection/port_availability",
            "INVALID",
            "unsupported port availability disposition",
        ),
    }

    match supply.get("reservation_status").and_then(Value::as_str) {
        Some("confirmed") => {}
        Some("provisional") | Some("unknown") | None => findings.push(
            "/port_supply/reservation_status",
            "INCOMPLETE",
            "port power reservation is not confirmed",
        ),
        Some("revoked") => findings.push(
            "/port_supply/reservation_status",
            "BLOCKED",
            "port power reservation has been revoked",
        ),
        _ => findings.push(
            "/port_supply/reservation_status",
            "INVALID",
            "unsupported reservation status",
        ),
    }

    let delivery_eligible = compatibility == Some("compatible")
        && connection.get("port_availability").and_then(Value::as_str) == Some("available")
        && supply.get("reservation_status").and_then(Value::as_str) == Some("confirmed")
        && usable_connection_seconds > 0
        && !findings.invalid
        && !findings.blocked
        && !findings.incomplete;

    let effective_connection_power_w = min_u64(&[
        requested_max_power_w,
        ship_max_power_w,
        berth_limit_w,
        reserved_power_w,
    ]);

    let power_time_energy_wh = (effective_connection_power_w as u128)
        .checked_mul(usable_connection_seconds as u128)
        .map(|value| value / 3_600)
        .and_then(|value| u64::try_from(value).ok());

    let power_time_energy_wh = match power_time_energy_wh {
        Some(value) => value,
        None => {
            findings.push(
                "/connection",
                "INVALID",
                "power-time energy calculation overflowed or could not be represented",
            );
            0
        }
    };

    let delivered_energy_wh = if delivery_eligible {
        min_u64(&[
            requested_energy_wh,
            available_energy_wh,
            power_time_energy_wh,
        ])
    } else {
        0
    };
    let unserved_energy_wh = requested_energy_wh.saturating_sub(delivered_energy_wh);

    let ghg = case.get("ghg_comparison").unwrap_or(&Value::Null);
    let grid_factor = ghg.get("grid_factor").unwrap_or(&Value::Null);
    let generator_factor = ghg
        .get("onboard_generator_factor")
        .unwrap_or(&Value::Null);

    check_factor_source(
        grid_factor,
        "/ghg_comparison/grid_factor",
        &source_ids,
        &mut findings,
    );
    check_factor_source(
        generator_factor,
        "/ghg_comparison/onboard_generator_factor",
        &source_ids,
        &mut findings,
    );

    let grid_boundary = grid_factor.get("boundary_id").and_then(Value::as_str);
    let generator_boundary = generator_factor.get("boundary_id").and_then(Value::as_str);
    if grid_boundary.is_some() && generator_boundary.is_some() && grid_boundary != generator_boundary {
        findings.push(
            "/ghg_comparison",
            "INCOMPARABLE",
            "grid and onboard-generator GHG factors use different accounting boundaries",
        );
    }

    let grid_rate = grid_factor
        .get("micro_kg_co2e_per_kwh")
        .and_then(Value::as_u64);
    let generator_rate = generator_factor
        .get("micro_kg_co2e_per_kwh")
        .and_then(Value::as_u64);

    let mut grid_ghg = None;
    let mut generator_ghg = None;
    let mut net_ghg_delta = None;
    if grid_boundary.is_some() && grid_boundary == generator_boundary {
        if let (Some(grid_rate), Some(generator_rate)) = (grid_rate, generator_rate) {
            check_factor_source(
                grid_factor,
                "/ghg_comparison/grid_factor",
                &source_ids,
                &mut findings,
            );
            check_factor_source(
                generator_factor,
                "/ghg_comparison/onboard_generator_factor",
                &source_ids,
                &mut findings,
            );
            grid_ghg = ghg_for_energy(delivered_energy_wh, grid_rate);
            generator_ghg = ghg_for_energy(delivered_energy_wh, generator_rate);
            match (grid_ghg, generator_ghg) {
                (Some(grid_value), Some(generator_value)) => {
                    match (
                        i128::try_from(grid_value),
                        i128::try_from(generator_value),
                    ) {
                        (Ok(grid_value), Ok(generator_value)) => {
                            net_ghg_delta = Some(grid_value - generator_value);
                        }
                        _ => findings.push(
                            "/ghg_comparison",
                            "INVALID",
                            "GHG arithmetic result exceeds signed 128-bit reporting range",
                        ),
                    }
                }
                _ => findings.push(
                    "/ghg_comparison",
                    "INVALID",
                    "GHG factor multiplication overflowed",
                ),
            }
        } else {
            findings.push(
                "/ghg_comparison",
                "INCOMPLETE",
                "grid and generator emission factors must be known before calculating a delta",
            );
        }
    }

    AuditReport {
        auditor: "marine-port-call-energy-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        scenario_id: case.get("scenario_id").and_then(Value::as_str).map(str::to_owned),
        port_call_id: case.get("port_call_id").and_then(Value::as_str).map(str::to_owned),
        record_status: case.get("record_status").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Deterministic declared OPS energy-cap and comparable-factor arithmetic only. Not grid/port availability proof, OPS electrical certification, legal applicability, physical savings, FuelEU/EU ETS verification, or operational authority.",
        delivery_eligible,
        usable_connection_seconds,
        effective_connection_power_w,
        requested_energy_wh,
        delivered_energy_wh,
        unserved_energy_wh,
        grid_ghg_micro_kg_co2e: grid_ghg,
        displaced_generator_ghg_micro_kg_co2e: generator_ghg,
        net_ghg_delta_micro_kg_co2e: net_ghg_delta,
        diagnostics: findings.diagnostics,
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(
            "usage: marine-port-call-energy-audit <schema.json> <case.json>".to_owned(),
        );
    }
    let schema = read_json(&args[1])?;
    let case = read_json(&args[2])?;
    let report = run_audit(&schema, &case);
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "SCENARIO_ACCOUNTING_VALID" { 0 } else { 2 })
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

    const PORT_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-port-call-energy-v0.1.0.schema.json");
    const PORT_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-port-call-energy-v0.1.0.json");

    fn fixture() -> (Value, Value) {
        (
            serde_json::from_str(PORT_SCHEMA).expect("port schema JSON"),
            serde_json::from_str(PORT_FIXTURE).expect("port-call fixture JSON"),
        )
    }

    #[test]
    fn known_answer_port_energy_and_ghg_arithmetic() {
        let (schema, case) = fixture();
        let report = run_audit(&schema, &case);
        assert_eq!(report.usable_connection_seconds, 20_400);
        assert_eq!(report.effective_connection_power_w, 300_000);
        assert_eq!(report.requested_energy_wh, 2_000_000);
        assert_eq!(report.delivered_energy_wh, 1_700_000);
        assert_eq!(report.unserved_energy_wh, 300_000);
        assert_eq!(report.grid_ghg_micro_kg_co2e, Some(340_000_000));
        assert_eq!(report.displaced_generator_ghg_micro_kg_co2e, Some(1_360_000_000));
        assert_eq!(report.net_ghg_delta_micro_kg_co2e, Some(-1_020_000_000));
        assert_eq!(report.disposition, "INCOMPLETE"); // synthetic record is never operational evidence
    }

    #[test]
    fn schema_fixture_pair_is_structurally_valid() {
        let (schema, case) = fixture();
        let mut findings = Findings::default();
        validate_schema(&schema, &case, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn incompatible_connector_blocks_delivery() {
        let (schema, mut case) = fixture();
        case["connection"]["compatibility"] = serde_json::json!("incompatible");
        let report = run_audit(&schema, &case);
        assert_eq!(report.delivered_energy_wh, 0);
        assert_eq!(report.unserved_energy_wh, 2_000_000);
        assert_eq!(report.disposition, "BLOCKED");
    }

    #[test]
    fn unknown_compatibility_is_incomplete_not_eligible() {
        let (schema, mut case) = fixture();
        case["connection"]["compatibility"] = serde_json::json!("unknown");
        let report = run_audit(&schema, &case);
        assert!(!report.delivery_eligible);
        assert_eq!(report.delivered_energy_wh, 0);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn connection_overhead_cannot_exceed_berth_stay() {
        let (schema, mut case) = fixture();
        case["berth_window"]["connection_time_seconds"] = serde_json::json!(30_000);
        case["berth_window"]["disconnection_time_seconds"] = serde_json::json!(30_000);
        let report = run_audit(&schema, &case);
        assert_eq!(report.usable_connection_seconds, 0);
        assert_eq!(report.delivered_energy_wh, 0);
        assert_eq!(report.disposition, "BLOCKED");
    }

    #[test]
    fn grid_generator_boundary_mismatch_prevents_net_ghg_comparison() {
        let (schema, mut case) = fixture();
        case["ghg_comparison"]["onboard_generator_factor"]["boundary_id"] =
            serde_json::json!("DIFFERENT-BOUNDARY");
        let report = run_audit(&schema, &case);
        assert_eq!(report.net_ghg_delta_micro_kg_co2e, None);
        assert_eq!(report.disposition, "INCOMPLETE"); // synthetic record precedence
        assert!(report.diagnostics.iter().any(|diagnostic| diagnostic.code == "INCOMPARABLE"));
    }

    #[test]
    fn unresolved_grid_factor_source_is_invalid() {
        let (schema, mut case) = fixture();
        case["ghg_comparison"]["grid_factor"]["source_id"] =
            serde_json::json!("MISSING-FACTOR-SOURCE");
        let report = run_audit(&schema, &case);
        assert_eq!(report.disposition, "INVALID");
    }

    #[test]
    fn revoked_reservation_blocks_energy_delivery() {
        let (schema, mut case) = fixture();
        case["port_supply"]["reservation_status"] = serde_json::json!("revoked");
        let report = run_audit(&schema, &case);
        assert_eq!(report.delivered_energy_wh, 0);
        assert_eq!(report.disposition, "BLOCKED");
    }

    #[test]
    fn zero_reserved_power_is_known_zero_not_unknown() {
        let (schema, mut case) = fixture();
        case["port_supply"]["reserved_power_w"] = serde_json::json!(0);
        let report = run_audit(&schema, &case);
        assert_eq!(report.effective_connection_power_w, 0);
        assert_eq!(report.delivered_energy_wh, 0);
        assert!(report.delivery_eligible);
    }

    #[test]
    fn port_energy_limit_caps_delivery_independently_of_power() {
        let (schema, mut case) = fixture();
        case["port_supply"]["available_energy_wh"] = serde_json::json!(500_000);
        let report = run_audit(&schema, &case);
        assert_eq!(report.delivered_energy_wh, 500_000);
        assert_eq!(report.unserved_energy_wh, 1_500_000);
    }
}
