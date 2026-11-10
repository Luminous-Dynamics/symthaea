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

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    record_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    blocked: bool,
    incomplete: bool,
    research_only: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
            "BLOCKED" => self.blocked = true,
            "INCOMPLETE" => self.incomplete = true,
            "RESEARCH_ONLY" => self.research_only = true,
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
        } else if self.incomplete {
            "NOT_READY"
        } else if self.research_only {
            "RESEARCH_ONLY"
        } else {
            "READY_FOR_FUTURE_METHOD_REVIEW"
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
        findings.push("/", "INVALID", "wind-energy evidence schema does not validate against Draft 2020-12 meta-schema");
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
            format!("wind-energy evidence schema violation: {error}"),
        );
    }
}

fn source_ids(record: &Value) -> BTreeSet<String> {
    record.get("evidence_source_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn require_source(
    record: &Value,
    sources: &BTreeSet<String>,
    path: &str,
    findings: &mut Findings,
) {
    match record.get("evidence_source_id").and_then(Value::as_str) {
        Some(id) if sources.contains(id) => {}
        Some(id) => findings.push(
            format!("{path}/evidence_source_id"),
            "INVALID",
            format!("evidence source '{id}' does not resolve in /evidence_source_ids"),
        ),
        None => findings.push(
            format!("{path}/evidence_source_id"),
            "INVALID",
            "energy quantity must resolve an evidence source",
        ),
    }
}

fn check_energy_quantity(
    quantity: &Value,
    path: &str,
    sources: &BTreeSet<String>,
    findings: &mut Findings,
) {
    if quantity.get("unit").and_then(Value::as_str) != Some("GJ") {
        findings.push(format!("{path}/unit"), "INVALID", "wind/onboard energy quantities in this evidence contract must use GJ");
    }
    require_source(quantity, sources, path, findings);
    if let Some(value) = quantity.get("value") {
        if !value.is_null() && value.as_f64().is_none_or(|number| !number.is_finite() || number < 0.0) {
            findings.push(format!("{path}/value"), "INVALID", "energy amount must be finite and non-negative or explicitly unknown");
        }
    }
}

fn check_record(record: &Value, findings: &mut Findings) {
    let status = record.get("record_status").and_then(Value::as_str).unwrap_or("");
    let sources = source_ids(record);
    let proposal_status = record.pointer("/proposal_basis/legal_status").and_then(Value::as_str);
    if proposal_status != Some("proposal_not_in_force") {
        findings.push("/proposal_basis/legal_status", "INVALID", "this v0.1 contract records COM(2026)620 as a proposal, not in-force law");
    }
    if record.pointer("/proposal_basis/instrument").and_then(Value::as_str) != Some("COM(2026)620")
        || record.pointer("/proposal_basis/procedure_id").and_then(Value::as_str) != Some("2026/0210(COD)")
    {
        findings.push("/proposal_basis", "INVALID", "proposal basis must bind COM(2026)620 and procedure 2026/0210(COD)");
    }
    if status == "synthetic" {
        findings.push("/record_status", "INCOMPLETE", "synthetic evidence fixture cannot establish a ship measurement or regulatory quantity");
    }

    let channels: BTreeSet<String> = record.get("observation_channel_ids")
        .and_then(Value::as_array)
        .into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect();
    let mut device_ids = BTreeSet::new();
    if let Some(devices) = record.get("devices").and_then(Value::as_array) {
        for (index, device) in devices.iter().enumerate() {
            let path = format!("/devices/{index}");
            if let Some(id) = device.get("device_id").and_then(Value::as_str) {
                if !device_ids.insert(id.to_owned()) {
                    findings.push(format!("{path}/device_id"), "INVALID", format!("duplicate WAPS device id '{id}'"));
                }
            }
            let commanded = device.get("commanded_state").and_then(Value::as_str);
            let achieved = device.get("achieved_state").and_then(Value::as_str);
            if commanded == Some("unknown") || achieved == Some("unknown") {
                findings.push(path.clone(), "INCOMPLETE", "WAPS commanded/achieved operating state is unknown");
            }
            if let Some(device_channels) = device.get("evidence_channel_ids").and_then(Value::as_array) {
                for (channel_index, channel) in device_channels.iter().enumerate() {
                    if let Some(channel) = channel.as_str() {
                        if !channels.contains(channel) {
                            findings.push(
                                format!("{path}/evidence_channel_ids/{channel_index}"),
                                "INVALID",
                                format!("device measurement channel '{channel}' does not resolve in observation_channel_ids"),
                            );
                        }
                    }
                }
            }
        }
    }

    let mut energy_source_ids = BTreeSet::new();
    let mut wind_source_count = 0usize;
    if let Some(energy_sources) = record.get("energy_sources").and_then(Value::as_array) {
        for (index, source) in energy_sources.iter().enumerate() {
            let path = format!("/energy_sources/{index}");
            if let Some(id) = source.get("energy_source_id").and_then(Value::as_str) {
                if !energy_source_ids.insert(id.to_owned()) {
                    findings.push(format!("{path}/energy_source_id"), "INVALID", format!("duplicate energy source id '{id}'"));
                }
            }
            if source.get("source_type").and_then(Value::as_str) == Some("wind_assistance") {
                wind_source_count += 1;
            }
            if let Some(quantity) = source.get("energy_gj") {
                check_energy_quantity(quantity, &format!("{path}/energy_gj"), &sources, findings);
            }
        }
    }

    let wind = record.get("wind_energy_assessment").unwrap_or(&Value::Null);
    let wind_status = wind.get("status").and_then(Value::as_str).unwrap_or("");
    match status {
        "certified" if wind_status != "certified" => findings.push(
            "/record_status",
            "INVALID",
            "record_status certified requires a certified wind-energy assessment",
        ),
        "method_derived" if wind_status != "standard_derived" && wind_status != "certified" => findings.push(
            "/record_status",
            "INVALID",
            "record_status method_derived requires a standard-derived or certified wind-energy assessment",
        ),
        _ => {}
    }
    let method_basis = wind.get("method_basis").and_then(Value::as_str).unwrap_or("");
    if status == "research_record" && wind_status == "certified" {
        findings.push("/record_status", "INVALID", "research_record cannot claim a certified wind-energy assessment");
    }
    if record.get("research_wind_contribution").is_some() {
        let research = record.get("research_wind_contribution").unwrap();
        check_energy_quantity(&research["energy_gj"], "/research_wind_contribution/energy_gj", &sources, findings);
        findings.push(
            "/research_wind_contribution",
            "RESEARCH_ONLY",
            "engineering wind-contribution estimate is distinct from any regulatory wind-energy quantity",
        );
    }

    let regulatory_quantity = wind.get("regulatory_energy_gj").unwrap_or(&Value::Null);
    check_energy_quantity(regulatory_quantity, "/wind_energy_assessment/regulatory_energy_gj", &sources, findings);

    let linked_wind_source = wind.get("energy_source_id").and_then(Value::as_str);
    match linked_wind_source {
        Some(id) if energy_source_ids.contains(id) => {
            let source = record.get("energy_sources").and_then(Value::as_array)
                .and_then(|arr| arr.iter().find(|entry| entry.get("energy_source_id").and_then(Value::as_str)==Some(id)));
            if source.and_then(|entry| entry.get("source_type")).and_then(Value::as_str) != Some("wind_assistance") {
                findings.push("/wind_energy_assessment/energy_source_id", "INVALID", "wind assessment must link to a wind_assistance energy source");
            }
        }
        Some(id) => findings.push("/wind_energy_assessment/energy_source_id", "INVALID", format!("wind energy source '{id}' does not resolve in /energy_sources")),
        None => findings.push("/wind_energy_assessment/energy_source_id", "INVALID", "wind assessment must identify its energy source ledger entry"),
    }
    if wind_source_count != 1 {
        findings.push("/energy_sources", "INVALID", "v0.1 requires exactly one wind_assistance energy-source entry per wind-energy assessment");
    }

    match wind_status {
        "not_determined" => {
            if regulatory_quantity.get("value").is_none_or(|value| !value.is_null()) {
                findings.push(
                    "/wind_energy_assessment/regulatory_energy_gj/value",
                    "INVALID",
                    "not_determined wind energy must remain explicit unknown; zero is not a substitute",
                );
            }
            if method_basis != "none" {
                findings.push("/wind_energy_assessment/method_basis", "INVALID", "not_determined status requires method_basis none");
            }
            if let Some(wind_id) = linked_wind_source {
                if let Some(source) = record.get("energy_sources").and_then(Value::as_array)
                    .and_then(|arr| arr.iter().find(|entry| entry.get("energy_source_id").and_then(Value::as_str)==Some(wind_id)))
                {
                    let amount = source.pointer("/energy_gj/value");
                    if amount.is_none_or(|value| !value.is_null()) {
                        findings.push(
                            "/energy_sources",
                            "INVALID",
                            "wind energy source ledger must remain unknown while the regulatory wind amount is not_determined",
                        );
                    }
                }
            }
            if status == "method_derived" || status == "certified" {
                findings.push("/record_status", "INVALID", "record cannot be method_derived/certified while wind amount is not_determined");
            }
            findings.push("/wind_energy_assessment", "INCOMPLETE", "no applicable wind-energy standard/implementing-act method has determined the regulatory quantity");
        }
        "research_estimate" => {
            if method_basis != "research_method" {
                findings.push("/wind_energy_assessment/method_basis", "INVALID", "research estimate must use research_method basis");
            }
            if wind.get("standard_reference").is_some() || wind.get("implementing_act_reference").is_some()
                || wind.get("certificate_id").is_some() || wind.get("certificate_sha256").is_some()
            {
                findings.push("/wind_energy_assessment", "INVALID", "research estimate cannot carry standards/certification claims");
            }
            if record.get("research_wind_contribution").is_none() {
                findings.push("/research_wind_contribution", "INCOMPLETE", "research_estimate status requires a separate research_wind_contribution record");
            }
            if regulatory_quantity.get("value").is_none_or(|value| !value.is_null()) {
                findings.push("/wind_energy_assessment/regulatory_energy_gj", "INVALID", "research estimate must not populate the regulatory wind-energy field before a valid method applies");
            }
        }
        "standard_derived" | "certified" => {
            if method_basis != "implementing_act_standard" || wind.get("standard_reference").and_then(Value::as_str).is_none()
                || wind.get("implementing_act_reference").and_then(Value::as_str).is_none()
                || wind.get("method_revision").and_then(Value::as_str).is_none()
            {
                findings.push("/wind_energy_assessment", "INVALID", "standard-derived wind amount requires an implementing-act standard, reference and method revision");
            }
            if wind_status == "certified" {
                for field in ["certificate_id","certificate_sha256","certification_body","verifier_attestation_id"] {
                    if wind.get(field).and_then(Value::as_str).is_none() {
                        findings.push(format!("/wind_energy_assessment/{field}"), "INVALID", "certified status requires certificate body, digest and verifier attestation");
                    }
                }
            }
            findings.push("/wind_energy_assessment", "BLOCKED", "COM(2026)620 is still a proposal; no implementing-act wind-energy method is in force in this v0.1 snapshot");
        }
        _ => findings.push("/wind_energy_assessment/status", "INVALID", "unsupported wind energy determination status"),
    }

    if let Some(counterfactual) = record.get("counterfactual_avoided_propulsion_energy") {
        check_energy_quantity(&counterfactual["avoided_propulsion_energy_gj"], "/counterfactual_avoided_propulsion_energy/avoided_propulsion_energy_gj", &sources, findings);
        if counterfactual.get("identifiability_status").and_then(Value::as_str) != Some("identified_for_declared_population") {
            findings.push("/counterfactual_avoided_propulsion_energy", "INCOMPLETE", "avoided propulsion energy is not causally identified for a declared population; it must not be represented as regulatory wind energy");
        } else {
            findings.push("/counterfactual_avoided_propulsion_energy", "RESEARCH_ONLY", "even an identified avoided-propulsion-energy counterfactual is not the regulatory wind-energy-used quantity");
        }
    }

    for (path, quantity) in [
        ("/research_wind_contribution/energy_gj", record.pointer("/research_wind_contribution/energy_gj")),
        ("/counterfactual_avoided_propulsion_energy/avoided_propulsion_energy_gj", record.pointer("/counterfactual_avoided_propulsion_energy/avoided_propulsion_energy_gj")),
    ] {
        if let Some(quantity) = quantity {
            require_source(quantity, &sources, path, findings);
        }
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-wind-energy-evidence-audit <schema.json> <evidence.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let evidence = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &evidence, &mut findings);
    check_record(&evidence, &mut findings);

    let report = AuditReport {
        auditor: "marine-wind-energy-evidence-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        record_id: evidence.get("record_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "WAPS energy-evidence contract checks only. No statutory wind-energy method, certification, verifier statement, fuel saving, physical qualification or legal compliance is established.",
        diagnostics: findings.diagnostics,
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "READY_FOR_FUTURE_METHOD_REVIEW" { 0 } else { 2 })
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

    const WIND_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-wind-energy-evidence-v0.1.0.schema.json");
    const WIND_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-wind-energy-evidence-v0.1.0.json");

    fn fixture() -> Value {
        serde_json::from_str(WIND_FIXTURE).expect("wind evidence fixture JSON")
    }

    #[test]
    fn committed_wind_energy_fixture_matches_schema() {
        let schema: Value = serde_json::from_str(WIND_SCHEMA).expect("wind schema JSON");
        let value = fixture();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("wind schema compiles");
        assert!(validator.is_valid(&value));
    }

    #[test]
    fn synthetic_unknown_wind_amount_is_not_ready() {
        let mut findings = Findings::default();
        check_record(&fixture(), &mut findings);
        assert_eq!(findings.disposition(), "NOT_READY");
        assert!(findings.incomplete);
    }

    #[test]
    fn not_determined_cannot_be_silently_zero() {
        let mut value = fixture();
        value["wind_energy_assessment"]["regulatory_energy_gj"]["value"] = serde_json::json!(0);
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn research_estimate_cannot_be_labelled_certified() {
        let mut value = fixture();
        value["wind_energy_assessment"]["status"] = serde_json::json!("research_estimate");
        value["wind_energy_assessment"]["method_basis"] = serde_json::json!("research_method");
        value["record_status"] = serde_json::json!("certified");
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn record_cannot_claim_certified_when_wind_assessment_is_research_only() {
        let mut value = fixture();
        value["record_status"] = serde_json::json!("certified");
        value["wind_energy_assessment"]["status"] = serde_json::json!("research_estimate");
        value["wind_energy_assessment"]["method_basis"] = serde_json::json!("research_method");
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn certified_wind_amount_is_blocked_while_proposal_is_pending() {
        let mut value = fixture();
        value["wind_energy_assessment"]["status"] = serde_json::json!("certified");
        value["wind_energy_assessment"]["method_basis"] = serde_json::json!("implementing_act_standard");
        value["wind_energy_assessment"]["standard_reference"] = serde_json::json!("SYNTHETIC-STANDARD");
        value["wind_energy_assessment"]["implementing_act_reference"] = serde_json::json!("SYNTHETIC-ACT");
        value["wind_energy_assessment"]["method_revision"] = serde_json::json!("1");
        value["wind_energy_assessment"]["certificate_id"] = serde_json::json!("SYNTHETIC-CERT");
        value["wind_energy_assessment"]["certificate_sha256"] = serde_json::json!("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
        value["wind_energy_assessment"]["certification_body"] = serde_json::json!("SYNTHETIC-BODY");
        value["wind_energy_assessment"]["verifier_attestation_id"] = serde_json::json!("SYNTHETIC-ATTESTATION");
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.blocked);
    }

    #[test]
    fn device_evidence_channels_must_resolve() {
        let mut value = fixture();
        value["devices"][0]["evidence_channel_ids"][0] = serde_json::json!("MISSING-CHANNEL");
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn wind_energy_source_must_remain_unknown_when_regulatory_amount_is_not_determined() {
        let mut value = fixture();
        value["energy_sources"][1]["energy_gj"]["value"] = serde_json::json!(3);
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn avoided_propulsion_energy_is_not_a_wind_energy_quantity() {
        let mut value = fixture();
        value["counterfactual_avoided_propulsion_energy"]["identifiability_status"] =
            serde_json::json!("identified_for_declared_population");
        let mut findings = Findings::default();
        check_record(&value, &mut findings);
        assert!(findings.incomplete || findings.research_only);
        assert_ne!(
            value["counterfactual_avoided_propulsion_energy"]["avoided_propulsion_energy_gj"]["value"],
            value["wind_energy_assessment"]["regulatory_energy_gj"]["value"]
        );
    }
}
