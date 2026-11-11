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
struct ComponentBalance {
    component_id: String,
    incoming_micro_gj: Option<i128>,
    outgoing_micro_gj: Option<i128>,
    loss_micro_gj: Option<i128>,
    storage_delta_micro_gj: Option<i128>,
    residual_micro_gj: Option<i128>,
    expanded_uncertainty_micro_gj: Option<u64>,
    disposition: &'static str,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    balance_id: Option<String>,
    disposition: &'static str,
    energy_unit: &'static str,
    external_input_micro_gj: Option<i128>,
    external_output_micro_gj: Option<i128>,
    total_loss_micro_gj: Option<i128>,
    total_storage_delta_micro_gj: Option<i128>,
    system_residual_micro_gj: Option<i128>,
    system_expanded_uncertainty_micro_gj: Option<u128>,
    component_balances: Vec<ComponentBalance>,
    diagnostics: Vec<Diagnostic>,
    claim_ceiling: &'static str,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    unbalanced: bool,
    incomparable: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
            "UNBALANCED" => self.unbalanced = true,
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
        } else if self.unbalanced {
            "UNBALANCED"
        } else if self.incomparable {
            "INCOMPARABLE"
        } else if self.incomplete {
            "INCOMPLETE"
        } else {
            "BALANCED_WITHIN_DECLARED_UNCERTAINTY"
        }
    }
}

#[derive(Clone, Copy)]
struct KnownEnergy {
    value: i128,
}

#[derive(Clone, Default)]
struct ComponentAccum {
    incoming: i128,
    outgoing: i128,
    unknown_incoming: bool,
    unknown_outgoing: bool,
    has_incoming: bool,
    has_outgoing: bool,
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
        findings.push("/", "INVALID", "energy-conservation schema does not validate against Draft 2020-12 meta-schema");
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
            format!("energy-conservation schema violation: {error}"),
        );
    }
}

fn add_i128(total: &mut i128, value: i128, path: &str, findings: &mut Findings) {
    match total.checked_add(value) {
        Some(sum) => *total = sum,
        None => findings.push(path, "INVALID", "signed integer energy accumulation overflowed"),
    }
}

fn add_u128(total: &mut u128, value: u128, path: &str, findings: &mut Findings) {
    match total.checked_add(value) {
        Some(sum) => *total = sum,
        None => findings.push(path, "INVALID", "uncertainty accumulation overflowed"),
    }
}

fn read_energy(
    value: &Value,
    path: &str,
    evidence_ids: &BTreeSet<String>,
    findings: &mut Findings,
) -> Option<KnownEnergy> {
    match value.get("value_micro_gj").and_then(Value::as_u64) {
        Some(amount) => {
            match value.get("evidence_source_id").and_then(Value::as_str) {
                Some(id) if evidence_ids.contains(id) => {}
                Some(id) => findings.push(
                    format!("{path}/evidence_source_id"),
                    "INVALID",
                    format!("evidence source '{id}' does not resolve"),
                ),
                None => findings.push(
                    format!("{path}/evidence_source_id"),
                    "INVALID",
                    "evidence source is required",
                ),
            }
            if value.get("method_id").and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
                findings.push(
                    format!("{path}/method_id"),
                    "INVALID",
                    "method_id is required",
                );
            }
            Some(KnownEnergy { value: amount as i128 })
        }
        _ if value.get("value_micro_gj").is_some_and(Value::is_null) => {
            if value.get("unknown_reason").and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
                findings.push(
                    format!("{path}/unknown_reason"),
                    "INVALID",
                    "unknown energy must retain an explicit reason",
                );
            } else {
                findings.push(
                    path,
                    "INCOMPLETE",
                    "energy is explicitly unknown and is not treated as zero",
                );
            }
            match value.get("evidence_source_id").and_then(Value::as_str) {
                Some(id) if evidence_ids.contains(id) => {}
                Some(id) => findings.push(
                    format!("{path}/evidence_source_id"),
                    "INVALID",
                    format!("evidence source '{id}' does not resolve"),
                ),
                None => findings.push(
                    format!("{path}/evidence_source_id"),
                    "INVALID",
                    "evidence source is required",
                ),
            }
            if value.get("method_id").and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
                findings.push(
                    format!("{path}/method_id"),
                    "INVALID",
                    "method_id is required even when the energy amount is unknown",
                );
            }
            None
        }
        _ => {
            findings.push(
                format!("{path}/value_micro_gj"),
                "INVALID",
                "energy must be a non-negative integer micro-GJ or explicit unknown",
            );
            None
        }
    }
}

fn read_storage_delta(
    value: &Value,
    path: &str,
    evidence_ids: &BTreeSet<String>,
    findings: &mut Findings,
) -> Option<i128> {
    match value.get("evidence_source_id").and_then(Value::as_str) {
        Some(id) if evidence_ids.contains(id) => {}
        Some(id) => findings.push(
            format!("{path}/evidence_source_id"),
            "INVALID",
            format!("storage-change evidence source '{id}' does not resolve"),
        ),
        None => findings.push(
            format!("{path}/evidence_source_id"),
            "INVALID",
            "storage-change evidence source is required",
        ),
    }
    if value.get("method_id").and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
        findings.push(format!("{path}/method_id"), "INVALID", "storage-change method_id is required");
    }

    let direction = value.get("direction").and_then(Value::as_str).unwrap_or("");
    match direction {
        "increase" | "decrease" | "none" => {
            let Some(amount) = value.get("value_micro_gj").and_then(Value::as_u64) else {
                findings.push(
                    format!("{path}/value_micro_gj"),
                    "INVALID",
                    "known storage change requires a non-negative integer magnitude",
                );
                return None;
            };
            match direction {
                "increase" => Some(amount as i128),
                "decrease" => Some(-(amount as i128)),
                "none" if amount == 0 => Some(0),
                "none" => {
                    findings.push(
                        path,
                        "INVALID",
                        "direction none requires a zero storage change",
                    );
                    None
                }
                _ => None,
            }
        }
        "unknown" => {
            if value.get("value_micro_gj").is_none_or(|v| !v.is_null()) {
                findings.push(
                    format!("{path}/value_micro_gj"),
                    "INVALID",
                    "unknown storage change must use an explicit null value",
                );
            }
            if value.get("unknown_reason").and_then(Value::as_str).is_none_or(|s| s.trim().is_empty()) {
                findings.push(
                    format!("{path}/unknown_reason"),
                    "INVALID",
                    "unknown storage change requires a reason",
                );
            } else {
                findings.push(
                    path,
                    "INCOMPLETE",
                    "storage change is unknown and is not treated as zero",
                );
            }
            None
        }
        _ => {
            findings.push(
                format!("{path}/direction"),
                "INVALID",
                "unsupported storage-change direction",
            );
            None
        }
    }
}

fn calculate(record: &Value) -> AuditReport {
    let mut findings = Findings::default();
    if record.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "synthetic balance is a software fixture, not measured vessel evidence",
        );
    }
    if record.get("accounting_basis").and_then(Value::as_str) != Some("first_law_component_boundary_v1") {
        findings.push(
            "/accounting_basis",
            "INVALID",
            "unsupported energy-conservation accounting basis",
        );
    }

    let evidence_ids: BTreeSet<String> = record
        .get("evidence_source_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();

    let window = record.get("reporting_window").unwrap_or(&Value::Null);
    let start = window.get("start").and_then(Value::as_str).and_then(|s| DateTime::parse_from_rfc3339(s).ok());
    let end = window.get("end").and_then(Value::as_str).and_then(|s| DateTime::parse_from_rfc3339(s).ok());
    if start.is_none() || end.is_none() || start >= end {
        findings.push(
            "/reporting_window",
            "INVALID",
            "reporting window must contain valid RFC3339 start < end",
        );
    }

    let components = record.get("components").and_then(Value::as_array).cloned().unwrap_or_default();
    let flows = record.get("flows").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut component_ids = BTreeSet::new();
    let mut component_by_id = BTreeMap::new();
    for (index, component) in components.iter().enumerate() {
        let path = format!("/components/{index}");
        let id = component.get("component_id").and_then(Value::as_str).unwrap_or("");
        if id.is_empty() || !component_ids.insert(id.to_owned()) {
            findings.push(
                format!("{path}/component_id"),
                "INVALID",
                format!("missing or duplicate component id '{id}'"),
            );
        } else {
            component_by_id.insert(id.to_owned(), component.clone());
        }
    }

    let mut flow_ids = BTreeSet::new();
    let mut measurement_ids = BTreeSet::new();
    let mut accum: BTreeMap<String, ComponentAccum> = component_ids
        .iter()
        .map(|id| (id.clone(), ComponentAccum::default()))
        .collect();

    let mut external_input: i128 = 0;
    let mut external_output: i128 = 0;
    let mut external_input_known = true;
    let mut external_output_known = true;
    let window_id = window.get("window_id").and_then(Value::as_str).unwrap_or("");

    for (index, flow) in flows.iter().enumerate() {
        let path = format!("/flows/{index}");
        let flow_id = flow.get("flow_id").and_then(Value::as_str).unwrap_or("");
        let measurement_id = flow.get("measurement_id").and_then(Value::as_str).unwrap_or("");

        if flow_id.is_empty() || !flow_ids.insert(flow_id.to_owned()) {
            findings.push(
                format!("{path}/flow_id"),
                "INVALID",
                format!("missing or duplicate flow id '{flow_id}'"),
            );
        }
        if measurement_id.is_empty() || !measurement_ids.insert(measurement_id.to_owned()) {
            findings.push(
                format!("{path}/measurement_id"),
                "INVALID",
                format!("missing or duplicate measurement id '{measurement_id}'"),
            );
        }
        if flow.get("reporting_window_id").and_then(Value::as_str) != Some(window_id) {
            findings.push(
                format!("{path}/reporting_window_id"),
                "INCOMPARABLE",
                "flow must bind the balance reporting window",
            );
        }

        let from = flow.get("from_component_id").and_then(Value::as_str);
        let to = flow.get("to_component_id").and_then(Value::as_str);
        let flow_class = flow.get("flow_class").and_then(Value::as_str).unwrap_or("");
        let endpoints_valid = match flow_class {
            "external_input" => {
                if flow.get("from_component_id").is_some_and(|v| !v.is_null()) || to.is_none() {
                    findings.push(
                        &path,
                        "INVALID",
                        "external_input requires null from_component_id and a destination component",
                    );
                    false
                } else {
                    true
                }
            }
            "external_output" => {
                if from.is_none() || flow.get("to_component_id").is_some_and(|v| !v.is_null()) {
                    findings.push(
                        &path,
                        "INVALID",
                        "external_output requires a source component and null to_component_id",
                    );
                    false
                } else {
                    true
                }
            }
            "internal_transfer" => {
                if from.is_none() || to.is_none() || from == to {
                    findings.push(
                        &path,
                        "INVALID",
                        "internal_transfer requires two distinct component endpoints",
                    );
                    false
                } else {
                    true
                }
            }
            _ => {
                findings.push(
                    format!("{path}/flow_class"),
                    "INVALID",
                    "unsupported flow_class",
                );
                false
            }
        };

        if endpoints_valid {
            for endpoint in [from, to].into_iter().flatten() {
                if !component_ids.contains(endpoint) {
                    findings.push(
                        &path,
                        "INVALID",
                        format!("flow endpoint '{endpoint}' does not resolve to a component"),
                    );
                }
            }
        }

        let amount = flow
            .get("energy")
            .map(|value| read_energy(value, &format!("{path}/energy"), &evidence_ids, &mut findings))
            .flatten();

        if let Some(energy) = flow.get("energy") {
            if energy.get("method_id") != flow.get("method_id") {
                findings.push(
                    format!("{path}/method_id"),
                    "INVALID",
                    "flow method_id must match its energy quantity method_id",
                );
            }
        }

        match (flow_class, amount) {
            ("external_input", Some(value)) if endpoints_valid => {
                add_i128(&mut external_input, value.value, &path, &mut findings);
            }
            ("external_input", None) => external_input_known = false,
            ("external_output", Some(value)) if endpoints_valid => {
                add_i128(&mut external_output, value.value, &path, &mut findings);
            }
            ("external_output", None) => external_output_known = false,
            _ => {}
        }

        if endpoints_valid {
            if let Some(id) = to {
                if !component_ids.contains(id) {
                    continue;
                }
                if let Some(entry) = accum.get_mut(id) {
                    entry.has_incoming = true;
                    match (flow_class, amount) {
                        ("external_input", Some(value)) | ("internal_transfer", Some(value)) => {
                            add_i128(&mut entry.incoming, value.value, &path, &mut findings);
                        }
                        ("external_input", None) | ("internal_transfer", None) => {
                            entry.unknown_incoming = true;
                        }
                        _ => {}
                    }
                }
            }
            if let Some(id) = from {
                if !component_ids.contains(id) {
                    continue;
                }
                if let Some(entry) = accum.get_mut(id) {
                    entry.has_outgoing = true;
                    match (flow_class, amount) {
                        ("external_output", Some(value)) | ("internal_transfer", Some(value)) => {
                            add_i128(&mut entry.outgoing, value.value, &path, &mut findings);
                        }
                        ("external_output", None) | ("internal_transfer", None) => {
                            entry.unknown_outgoing = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let mut component_balances = Vec::new();
    let mut total_loss: i128 = 0;
    let mut total_storage_delta: i128 = 0;
    let mut total_tolerance: u128 = 0;
    let mut losses_known = true;
    let mut storage_known = true;
    let mut tolerances_qualifying = true;

    for component in &components {
        let id = component.get("component_id").and_then(Value::as_str).unwrap_or("<missing>");
        let path = format!("/components/{id}");
        let component_type = component.get("component_type").and_then(Value::as_str).unwrap_or("");
        let acc = accum.get(id).cloned().unwrap_or_default();

        let loss = component
            .get("conversion_loss")
            .map(|value| read_energy(value, &format!("{path}/conversion_loss"), &evidence_ids, &mut findings))
            .flatten();
        let storage = component
            .get("storage_change")
            .map(|value| read_storage_delta(value, &format!("{path}/storage_change"), &evidence_ids, &mut findings))
            .flatten();

        if let Some(loss) = loss {
            add_i128(&mut total_loss, loss.value, &path, &mut findings);
        } else {
            losses_known = false;
        }
        if let Some(storage) = storage {
            add_i128(&mut total_storage_delta, storage, &path, &mut findings);
        } else {
            storage_known = false;
        }

        if !matches!(component_type, "battery_storage" | "thermal_storage" | "hydrogen_storage") {
            if storage.is_some_and(|value| value != 0) {
                findings.push(
                    format!("{path}/storage_change"),
                    "INVALID",
                    "non-storage component must declare zero storage change",
                );
            }
        }

        let budget = component.get("uncertainty_budget").unwrap_or(&Value::Null);
        let basis = budget.get("basis").and_then(Value::as_str).unwrap_or("unknown");
        let propagation = budget.get("propagation_method").and_then(Value::as_str).unwrap_or("unknown");
        let tolerance = budget.get("expanded_uncertainty_micro_gj").and_then(Value::as_u64);
        let mut tolerance_qualifies =
            matches!(basis, "metrology_expanded_uncertainty" | "deterministic_rounding")
                && !matches!(propagation, "unknown");

        match basis {
            "engineering_allowance" => {
                findings.push(
                    format!("{path}/uncertainty_budget/basis"),
                    "INCOMPLETE",
                    "engineering allowance is not a metrology uncertainty budget and cannot qualify balance closure",
                );
                tolerance_qualifies = false;
            }
            "unknown" => {
                findings.push(
                    format!("{path}/uncertainty_budget/basis"),
                    "INCOMPLETE",
                    "uncertainty-budget basis is unknown",
                );
                tolerance_qualifies = false;
            }
            _ => {}
        }
        if matches!(propagation, "unknown") || tolerance.is_none() {
            findings.push(
                format!("{path}/uncertainty_budget"),
                "INCOMPLETE",
                "a declared expanded uncertainty budget and propagation method are required",
            );
            tolerance_qualifies = false;
        }
        match budget.get("evidence_source_id").and_then(Value::as_str) {
            Some(source_id) if evidence_ids.contains(source_id) => {}
            Some(_) => {
                findings.push(
                    format!("{path}/uncertainty_budget/evidence_source_id"),
                    "INVALID",
                    "uncertainty-budget evidence source does not resolve",
                );
                tolerance_qualifies = false;
            }
            None => {
                findings.push(
                    format!("{path}/uncertainty_budget/evidence_source_id"),
                    "INVALID",
                    "uncertainty-budget evidence source is required",
                );
                tolerance_qualifies = false;
            }
        }
        if budget.get("coverage_factor").and_then(Value::as_f64).is_none_or(|value| value <= 0.0) {
            findings.push(
                format!("{path}/uncertainty_budget/coverage_factor"),
                "INVALID",
                "coverage factor must be positive",
            );
            tolerance_qualifies = false;
        }
        if budget.get("method_id").and_then(Value::as_str).is_none_or(|value| value.trim().is_empty()) {
            findings.push(
                format!("{path}/uncertainty_budget/method_id"),
                "INVALID",
                "uncertainty-budget method_id is required",
            );
            tolerance_qualifies = false;
        }

        if tolerance_qualifies {
            if let Some(tolerance) = tolerance {
                add_u128(&mut total_tolerance, tolerance as u128, &path, &mut findings);
            }
        } else {
            tolerances_qualifying = false;
        }

        let values_known = acc.has_incoming
            && acc.has_outgoing
            && !acc.unknown_incoming
            && !acc.unknown_outgoing
            && loss.is_some()
            && storage.is_some();

        let (residual, disposition) = if values_known && tolerance_qualifies {
            let residual = acc.incoming
                .checked_sub(acc.outgoing)
                .and_then(|value| value.checked_sub(loss.unwrap().value))
                .and_then(|value| value.checked_sub(storage.unwrap()));
            match residual {
                Some(residual) => {
                    let within = tolerance.is_some_and(|limit| residual.unsigned_abs() <= limit as u128);
                    if !within {
                        findings.push(
                            &path,
                            "UNBALANCED",
                            format!("component energy residual {residual} micro-GJ exceeds uncertainty budget {}", tolerance.unwrap()),
                        );
                    }
                    (Some(residual), if within { "WITHIN_DECLARED_UNCERTAINTY" } else { "UNBALANCED" })
                }
                None => {
                    findings.push(&path, "INVALID", "component residual arithmetic overflowed");
                    (None, "INVALID")
                }
            }
        } else {
            findings.push(
                &path,
                "INCOMPLETE",
                "component requires known incoming/outgoing flows, conversion loss, storage change and a qualifying uncertainty budget",
            );
            (None, "INCOMPLETE")
        };

        component_balances.push(ComponentBalance {
            component_id: id.to_owned(),
            incoming_micro_gj: if acc.unknown_incoming { None } else { Some(acc.incoming) },
            outgoing_micro_gj: if acc.unknown_outgoing { None } else { Some(acc.outgoing) },
            loss_micro_gj: loss.map(|value| value.value),
            storage_delta_micro_gj: storage,
            residual_micro_gj: residual,
            expanded_uncertainty_micro_gj: tolerance,
            disposition,
        });
    }

    let system_residual = if external_input_known && external_output_known && losses_known && storage_known {
        external_input
            .checked_sub(external_output)
            .and_then(|value| value.checked_sub(total_loss))
            .and_then(|value| value.checked_sub(total_storage_delta))
    } else {
        findings.push(
            "/",
            "INCOMPLETE",
            "whole-system boundary cannot be closed because one or more external flows, losses or storage changes are unknown",
        );
        None
    };
    if external_input_known && external_output_known && losses_known && storage_known && system_residual.is_none() {
        findings.push("/", "INVALID", "whole-system residual arithmetic overflowed");
    }

    let system_tolerance = if tolerances_qualifying { Some(total_tolerance) } else { None };
    if let (Some(residual), Some(tolerance)) = (system_residual, system_tolerance) {
        if residual.unsigned_abs() > tolerance {
            findings.push(
                "/",
                "UNBALANCED",
                format!("whole-system residual {residual} micro-GJ exceeds summed component uncertainty {tolerance}"),
            );
        }
    }

    AuditReport {
        auditor: "marine-energy-conservation-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        balance_id: record.get("balance_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        energy_unit: "micro-GJ (1 micro-GJ = 1,000 joules = 1 kJ)",
        external_input_micro_gj: if external_input_known { Some(external_input) } else { None },
        external_output_micro_gj: if external_output_known { Some(external_output) } else { None },
        total_loss_micro_gj: if losses_known { Some(total_loss) } else { None },
        total_storage_delta_micro_gj: if storage_known { Some(total_storage_delta) } else { None },
        system_residual_micro_gj: system_residual,
        system_expanded_uncertainty_micro_gj: system_tolerance,
        component_balances,
        diagnostics: findings.diagnostics,
        claim_ceiling: "First-law arithmetic closure against supplied values and supplied expanded-uncertainty budgets only. This does not verify meters, uncertainty budgets, physical models, exergy, source factors, emissions, safety, vessel performance or regulatory compliance.",
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-energy-conservation-audit <schema.json> <balance.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let record = read_json(&args[2])?;
    let mut schema_findings = Findings::default();
    validate_schema(&schema, &record, &mut schema_findings);
    let mut report = calculate(&record);
    schema_findings.diagnostics.append(&mut report.diagnostics);
    let disposition = if schema_findings.invalid || report.disposition == "INVALID" {
        "INVALID"
    } else if schema_findings.unbalanced || report.disposition == "UNBALANCED" {
        "UNBALANCED"
    } else if schema_findings.incomparable || report.disposition == "INCOMPARABLE" {
        "INCOMPARABLE"
    } else if schema_findings.incomplete || report.disposition == "INCOMPLETE" {
        "INCOMPLETE"
    } else {
        report.disposition
    };
    report.disposition = disposition;
    report.diagnostics = schema_findings.diagnostics;
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if disposition == "BALANCED_WITHIN_DECLARED_UNCERTAINTY" { 0 } else { 2 })
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

    const BALANCE_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-energy-conservation-v0.1.0.schema.json");
    const BALANCED_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-energy-conservation-v0.1.0.json");
    const UNBALANCED_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-energy-conservation-unbalanced-v0.1.0.json");

    fn parsed(source: &str) -> Value {
        serde_json::from_str(source).expect("energy conservation fixture JSON")
    }

    #[test]
    fn micro_gj_is_one_kilojoule_not_one_joule() {
        assert_eq!(1.0e-6 * 1.0e9, 1_000.0);
    }

    #[test]
    fn committed_schema_and_balanced_fixture_validate() {
        let schema: Value = serde_json::from_str(BALANCE_SCHEMA).expect("balance schema JSON");
        let record = parsed(BALANCED_FIXTURE);
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("schema compiles");
        assert!(validator.is_valid(&record));
    }

    #[test]
    fn integer_component_and_system_balances_close_exactly() {
        let report = calculate(&parsed(BALANCED_FIXTURE));
        assert_eq!(report.external_input_micro_gj, Some(110));
        assert_eq!(report.external_output_micro_gj, Some(25));
        assert_eq!(report.total_loss_micro_gj, Some(82));
        assert_eq!(report.total_storage_delta_micro_gj, Some(3));
        assert_eq!(report.system_residual_micro_gj, Some(0));
        assert!(report.component_balances.iter().all(|component| component.residual_micro_gj == Some(0)));
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn unbalanced_component_is_not_hidden_by_system_aggregate() {
        let report = calculate(&parsed(UNBALANCED_FIXTURE));
        assert_eq!(report.disposition, "UNBALANCED");
        assert!(report.diagnostics.iter().any(|diagnostic| diagnostic.code == "UNBALANCED"));
    }

    #[test]
    fn unknown_energy_flow_is_not_treated_as_zero() {
        let mut record = parsed(BALANCED_FIXTURE);
        record["flows"][0]["energy"]["value_micro_gj"] = Value::Null;
        record["flows"][0]["energy"]["unknown_reason"] = serde_json::json!("fuel measurement unavailable");
        let report = calculate(&record);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert_eq!(report.external_input_micro_gj, None);
    }

    #[test]
    fn duplicate_flow_identity_is_invalid() {
        let mut record = parsed(BALANCED_FIXTURE);
        record["flows"][1]["flow_id"] = record["flows"][0]["flow_id"].clone();
        let report = calculate(&record);
        assert_eq!(report.disposition, "INVALID");
    }

    #[test]
    fn unresolved_component_endpoint_is_invalid() {
        let mut record = parsed(BALANCED_FIXTURE);
        record["flows"][1]["to_component_id"] = serde_json::json!("MISSING-COMPONENT");
        let report = calculate(&record);
        assert_eq!(report.disposition, "INVALID");
    }

    #[test]
    fn non_storage_component_cannot_accumulate_energy() {
        let mut record = parsed(BALANCED_FIXTURE);
        record["components"][0]["storage_change"]["direction"] = serde_json::json!("increase");
        record["components"][0]["storage_change"]["value_micro_gj"] = serde_json::json!(1);
        let report = calculate(&record);
        assert_eq!(report.disposition, "INVALID");
    }

    #[test]
    fn engineering_allowance_does_not_qualify_as_metrology_uncertainty() {
        let mut record = parsed(BALANCED_FIXTURE);
        record["components"][0]["uncertainty_budget"]["basis"] = serde_json::json!("engineering_allowance");
        let report = calculate(&record);
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn internal_transfers_cancel_from_system_boundary_total() {
        let report = calculate(&parsed(BALANCED_FIXTURE));
        let residual = report.external_input_micro_gj.unwrap()
            - report.external_output_micro_gj.unwrap()
            - report.total_loss_micro_gj.unwrap()
            - report.total_storage_delta_micro_gj.unwrap();
        assert_eq!(residual, 0);
        assert_eq!(report.external_input_micro_gj, Some(110));
        assert_ne!(report.external_input_micro_gj, Some(150)); // internal shaft/battery flows are not added again
    }
}
