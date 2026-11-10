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
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    ledger_id: Option<String>,
    disposition: &'static str,
    accounting_basis: &'static str,
    energy_unit: &'static str,
    known_subtotal_micro_gj: u128,
    total_source_input_micro_gj: Option<u128>,
    included_source_ids: Vec<String>,
    non_additive_flow_ids: Vec<String>,
    unknown_input_ids: Vec<String>,
    diagnostics: Vec<Diagnostic>,
    claim_ceiling: &'static str,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
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
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "SOURCE_INPUT_TOTAL_VALID"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn validate_schema(schema: &Value, instance: &Value, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push("/$schema", "INVALID", "only Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push("/", "INVALID", "energy-source ledger schema does not validate against Draft 2020-12 meta-schema");
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
            format!("energy-source ledger schema violation: {error}"),
        );
    }
}

fn expected_role(source_type: &str) -> Option<&'static str> {
    match source_type {
        "fuel_consumption" => Some("primary_input"),
        "shore_electricity" | "onboard_renewable" | "wind_assistance" => Some("substitute_input"),
        "generator_output" | "battery_discharge" | "shaft_output" => Some("conversion_output"),
        "conversion_loss" => Some("loss"),
        _ => None,
    }
}

fn has_cycle(node: &str, graph: &BTreeMap<String, Vec<String>>, visiting: &mut BTreeSet<String>, visited: &mut BTreeSet<String>) -> bool {
    if visiting.contains(node) {
        return true;
    }
    if visited.contains(node) {
        return false;
    }
    visiting.insert(node.to_owned());
    if let Some(parents) = graph.get(node) {
        for parent in parents {
            if has_cycle(parent, graph, visiting, visited) {
                return true;
            }
        }
    }
    visiting.remove(node);
    visited.insert(node.to_owned());
    false
}

struct Computation {
    findings: Findings,
    known_subtotal_micro_gj: u128,
    total_source_input_micro_gj: Option<u128>,
    included_source_ids: Vec<String>,
    non_additive_flow_ids: Vec<String>,
    unknown_input_ids: Vec<String>,
}

fn calculate(record: &Value) -> Computation {
    let mut findings = Findings::default();
    let mut sources = BTreeSet::new();
    let mut measurements = BTreeSet::new();
    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut known_subtotal: u128 = 0;
    let mut total_overflow = false;
    let mut input_complete = true;
    let mut included_ids = Vec::new();
    let mut non_additive_ids = Vec::new();
    let mut unknown_input_ids = Vec::new();
    let evidence_ids: BTreeSet<String> = record.get("evidence_source_ids")
        .and_then(Value::as_array).into_iter().flatten()
        .filter_map(Value::as_str).map(str::to_owned).collect();

    if record.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push("/record_status", "INCOMPLETE", "synthetic ledger is a software fixture, not measured vessel energy evidence");
    }
    if record.get("accounting_basis").and_then(Value::as_str) != Some("research_source_input_boundary_v1") {
        findings.push("/accounting_basis", "INVALID", "unsupported energy accounting boundary");
    }

    let entries = record.get("energy_sources").and_then(Value::as_array);
    let Some(entries) = entries else {
        findings.push("/energy_sources", "INVALID", "energy_sources must be an array");
        return Computation {
            findings,
            known_subtotal_micro_gj: 0,
            total_source_input_micro_gj: None,
            included_source_ids: included_ids,
            non_additive_flow_ids: non_additive_ids,
            unknown_input_ids,
        };
    };

    for (index, entry) in entries.iter().enumerate() {
        let path = format!("/energy_sources/{index}");
        let source_id = entry.get("source_id").and_then(Value::as_str).unwrap_or("");
        let measurement_id = entry.get("measurement_id").and_then(Value::as_str).unwrap_or("");
        let source_type = entry.get("source_type").and_then(Value::as_str).unwrap_or("");
        let flow_role = entry.get("flow_role").and_then(Value::as_str).unwrap_or("");

        if source_id.is_empty() {
            findings.push(format!("{path}/source_id"), "INVALID", "source_id is required");
        } else if !sources.insert(source_id.to_owned()) {
            findings.push(format!("{path}/source_id"), "INVALID", format!("duplicate source id '{source_id}'"));
        }

        if measurement_id.is_empty() {
            findings.push(format!("{path}/measurement_id"), "INVALID", "measurement_id is required");
        } else if !measurements.insert(measurement_id.to_owned()) {
            findings.push(format!("{path}/measurement_id"), "INVALID", format!("duplicate measurement id '{measurement_id}'"));
        }

        match expected_role(source_type) {
            Some(expected) if expected == flow_role => {}
            Some(expected) => findings.push(
                format!("{path}/flow_role"),
                "INVALID",
                format!("source_type '{source_type}' requires flow_role '{expected}', not '{flow_role}'"),
            ),
            None => findings.push(format!("{path}/source_type"), "INVALID", "unsupported energy source type"),
        }

        if let Some(quantity) = entry.get("energy") {
            if quantity.get("evidence_source_id").and_then(Value::as_str).is_none_or(|id| !evidence_ids.contains(id)) {
                findings.push(
                    format!("{path}/energy/evidence_source_id"),
                    "INVALID",
                    "energy evidence_source_id does not resolve in evidence_source_ids",
                );
            }

            let amount = quantity.get("value_micro_gj");
            let parents = entry.get("parent_source_ids").and_then(Value::as_array).cloned().unwrap_or_default();

            match flow_role {
                "primary_input" | "substitute_input" => {
                    if !parents.is_empty() {
                        findings.push(
                            format!("{path}/parent_source_ids"),
                            "INVALID",
                            "primary/substitute source inputs cannot have upstream parent_source_ids in this ledger",
                        );
                    }
                    match amount.and_then(Value::as_u64) {
                        Some(value) => {
                            known_subtotal = match known_subtotal.checked_add(value as u128) {
                                Some(total) => total,
                                None => {
                                    total_overflow = true;
                                    known_subtotal
                                }
                            };
                            included_ids.push(source_id.to_owned());
                        }
                        None if amount.is_some_and(Value::is_null) => {
                            input_complete = false;
                            unknown_input_ids.push(source_id.to_owned());
                            findings.push(
                                format!("{path}/energy"),
                                "INCOMPLETE",
                                "unknown primary/substitute source input prevents a complete source-input total; it is not zero",
                            );
                        }
                        _ => findings.push(format!("{path}/energy/value_micro_gj"), "INVALID", "source input must be a non-negative integer micro-GJ or explicit null with reason"),
                    }
                }
                "conversion_output" | "loss" => {
                    non_additive_ids.push(source_id.to_owned());
                    if parents.is_empty() {
                        findings.push(
                            format!("{path}/parent_source_ids"),
                            "INVALID",
                            "conversion outputs and losses must identify at least one parent input/source",
                        );
                    }
                    if amount.and_then(Value::as_u64).is_none() {
                        findings.push(
                            format!("{path}/energy/value_micro_gj"),
                            "INCOMPLETE",
                            "conversion-output/loss amount is not known; the flow record remains incomplete but is not added to source inputs",
                        );
                    }
                }
                _ => {}
            }

            graph.insert(source_id.to_owned(), parents.iter().filter_map(Value::as_str).map(str::to_owned).collect());
        } else {
            findings.push(format!("{path}/energy"), "INVALID", "energy quantity is missing");
        }
    }

    for (index, entry) in entries.iter().enumerate() {
        let path = format!("/energy_sources/{index}");
        let parents = entry.get("parent_source_ids").and_then(Value::as_array).cloned().unwrap_or_default();
        for (parent_index, parent) in parents.iter().enumerate() {
            match parent.as_str() {
                Some(id) if sources.contains(id) => {}
                Some(id) => findings.push(
                    format!("{path}/parent_source_ids/{parent_index}"),
                    "INVALID",
                    format!("parent source '{id}' does not resolve in the ledger"),
                ),
                None => findings.push(format!("{path}/parent_source_ids/{parent_index}"), "INVALID", "parent source ID must be a string"),
            }
        }
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for source_id in &sources {
        if has_cycle(source_id, &graph, &mut visiting, &mut visited) {
            findings.push("/energy_sources", "INVALID", "energy conversion dependency graph contains a cycle");
            break;
        }
    }

    if total_overflow {
        findings.push("/energy_sources", "INVALID", "source-input total overflowed the supported integer accumulator");
    }

    let total = if input_complete && !findings.invalid && !total_overflow {
        Some(known_subtotal)
    } else {
        None
    };

    Computation {
        findings,
        known_subtotal_micro_gj: known_subtotal,
        total_source_input_micro_gj: total,
        included_source_ids: included_ids,
        non_additive_flow_ids: non_additive_ids,
        unknown_input_ids,
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-energy-source-ledger-audit <schema.json> <ledger.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let ledger = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &ledger, &mut findings);
    let computation = calculate(&ledger);
    findings.diagnostics.extend(computation.findings.diagnostics.clone());
    findings.invalid |= computation.findings.invalid;
    findings.incomplete |= computation.findings.incomplete;

    let report = AuditReport {
        auditor: "marine-energy-source-ledger-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        ledger_id: ledger.get("ledger_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        accounting_basis: "research_source_input_boundary_v1",
        energy_unit: "micro_gj (1 micro_gj = 1 joule)",
        known_subtotal_micro_gj: computation.known_subtotal_micro_gj,
        total_source_input_micro_gj: computation.total_source_input_micro_gj,
        included_source_ids: computation.included_source_ids,
        non_additive_flow_ids: computation.non_additive_flow_ids,
        unknown_input_ids: computation.unknown_input_ids,
        diagnostics: findings.diagnostics,
        claim_ceiling: "Deterministic ledger/source-flow arithmetic only. Not a statutory energy formula, complete fuel/energy report, MRV/FuelEU verifier result, causal savings claim or physical qualification.",
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "SOURCE_INPUT_TOTAL_VALID" { 0 } else { 2 })
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

    const LEDGER_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-energy-source-ledger-v0.1.0.schema.json");
    const LEDGER_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-energy-source-ledger-v0.1.0.json");

    fn fixture() -> Value {
        serde_json::from_str(LEDGER_FIXTURE).expect("ledger fixture JSON")
    }

    #[test]
    fn committed_ledger_fixture_matches_schema() {
        let schema: Value = serde_json::from_str(LEDGER_SCHEMA).expect("ledger schema JSON");
        let value = fixture();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("ledger schema compiles");
        assert!(validator.is_valid(&value));
    }

    #[test]
    fn unknown_wind_input_returns_subtotal_but_no_complete_total() {
        let result = calculate(&fixture());
        assert_eq!(result.known_subtotal_micro_gj, 102_000_000);
        assert_eq!(result.total_source_input_micro_gj, None);
        assert_eq!(result.unknown_input_ids, vec!["WIND-ASSIST"]);
        assert!(result.findings.incomplete);
    }

    #[test]
    fn known_answer_total_excludes_generator_battery_and_shaft_outputs() {
        let mut value = fixture();
        value["energy_sources"][2]["energy"]["value_micro_gj"] = serde_json::json!(10_000_000);
        value["energy_sources"][2]["energy"].as_object_mut().unwrap().remove("unknown_reason");
        let result = calculate(&value);
        assert_eq!(result.known_subtotal_micro_gj, 112_000_000);
        assert_eq!(result.total_source_input_micro_gj, Some(112_000_000));
        assert_eq!(result.non_additive_flow_ids.len(), 4);
        assert!(!result.findings.invalid);
    }

    #[test]
    fn duplicate_source_identity_is_invalid() {
        let mut value = fixture();
        value["energy_sources"][1]["source_id"] = serde_json::json!("FUEL-MAIN");
        let result = calculate(&value);
        assert!(result.findings.invalid);
    }

    #[test]
    fn source_type_must_match_flow_role() {
        let mut value = fixture();
        value["energy_sources"][0]["flow_role"] = serde_json::json!("conversion_output");
        let result = calculate(&value);
        assert!(result.findings.invalid);
    }

    #[test]
    fn conversion_parent_must_resolve() {
        let mut value = fixture();
        value["energy_sources"][3]["parent_source_ids"] = serde_json::json!(["MISSING-PARENT"]);
        let result = calculate(&value);
        assert!(result.findings.invalid);
    }

    #[test]
    fn conversion_dependency_cycle_is_invalid() {
        let mut value = fixture();
        value["energy_sources"][0]["parent_source_ids"] = serde_json::json!(["GEN-OUTPUT"]);
        let result = calculate(&value);
        assert!(result.findings.invalid);
    }

    #[test]
    fn downstream_conversion_unknown_does_not_change_input_total_but_keeps_record_incomplete() {
        let mut value = fixture();
        value["energy_sources"][3]["energy"]["value_micro_gj"] = Value::Null;
        value["energy_sources"][3]["energy"]["unknown_reason"] = serde_json::json!("synthetic missing downstream meter");
        let result = calculate(&value);
        assert_eq!(result.known_subtotal_micro_gj, 102_000_000);
        assert_eq!(result.total_source_input_micro_gj, None); // unknown wind input still blocks the total
        assert!(result.findings.incomplete);
    }
}
