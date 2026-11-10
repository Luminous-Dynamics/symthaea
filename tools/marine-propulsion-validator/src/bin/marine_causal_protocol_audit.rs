use chrono::DateTime;
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
    protocol_id: Option<String>,
    readiness_disposition: &'static str,
    claim_ceiling: &'static str,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Default)]
struct Findings {
    diagnostics: Vec<Diagnostic>,
    invalid: bool,
    blocked: bool,
    incomplete: bool,
}

impl Findings {
    fn push(&mut self, path: impl Into<String>, code: &'static str, message: impl Into<String>) {
        match code {
            "INVALID" => self.invalid = true,
            "BLOCKED" => self.blocked = true,
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
        } else if self.incomplete {
            "NOT_READY"
        } else {
            "READY_FOR_INDEPENDENT_REVIEW"
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
        findings.push("/", "INVALID", "schema does not validate against the Draft 2020-12 meta-schema");
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
            format!("causal protocol schema violation: {error}"),
        );
    }
}

fn valid_rfc3339(value: Option<&str>) -> Option<chrono::DateTime<chrono::FixedOffset>> {
    value.and_then(|text| DateTime::parse_from_rfc3339(text).ok())
}

fn check_protocol(protocol: &Value, findings: &mut Findings) {
    if protocol.get("schema_version").and_then(Value::as_str) != Some("0.1.0") {
        findings.push("/schema_version", "INVALID", "unsupported causal protocol version");
    }
    if protocol.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push("/record_status", "INCOMPLETE", "synthetic protocol fixture cannot establish causal identification");
    }

    let study = protocol.get("study").unwrap_or(&Value::Null);
    let start = valid_rfc3339(study.get("study_start").and_then(Value::as_str));
    let end = valid_rfc3339(study.get("study_end").and_then(Value::as_str));
    let preregistered = valid_rfc3339(study.get("preregistered_at").and_then(Value::as_str));
    match (preregistered, start, end) {
        (Some(preregistered), Some(start), Some(end)) if start < end && preregistered < start => {}
        (Some(_), Some(start), Some(end)) if start >= end => {
            findings.push("/study/study_end", "INVALID", "study_end must be later than study_start");
        }
        (Some(_), Some(_), Some(_)) => {
            findings.push("/study/preregistered_at", "INVALID", "protocol must be preregistered before study_start");
        }
        _ => findings.push("/study", "INVALID", "study dates must be valid RFC3339 timestamps"),
    }

    let outcome_ids: BTreeSet<String> = protocol.get("outcomes").and_then(Value::as_array)
        .into_iter().flatten()
        .filter_map(|outcome| outcome.get("outcome_id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    if !outcome_ids.contains(
        protocol.pointer("/estimand/outcome_id").and_then(Value::as_str).unwrap_or("")
    ) {
        findings.push("/estimand/outcome_id", "INVALID", "estimand outcome_id does not resolve in the outcome registry");
    }
    let mut unique_outcome_ids = BTreeSet::new();
    if let Some(outcomes) = protocol.get("outcomes").and_then(Value::as_array) {
        for (index, outcome) in outcomes.iter().enumerate() {
            if let Some(id) = outcome.get("outcome_id").and_then(Value::as_str) {
                if !unique_outcome_ids.insert(id.to_owned()) {
                    findings.push(format!("/outcomes/{index}/outcome_id"), "INVALID", format!("duplicate outcome id '{id}'"));
                }
            }
        }
    }

    let source_ids: BTreeSet<String> = protocol.get("evidence_source_ids").and_then(Value::as_array)
        .into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect();
    let mut confounder_ids = BTreeSet::new();
    let mut adjustment_ids = BTreeSet::new();
    if let Some(confounders) = protocol.get("confounders").and_then(Value::as_array) {
        for (index, confounder) in confounders.iter().enumerate() {
            let path = format!("/confounders/{index}");
            if let Some(id) = confounder.get("confounder_id").and_then(Value::as_str) {
                if !confounder_ids.insert(id.to_owned()) {
                    findings.push(format!("{path}/confounder_id"), "INVALID", format!("duplicate confounder id '{id}'"));
                }
            }
            let role = confounder.get("causal_role").and_then(Value::as_str).unwrap_or("");
            let method = confounder.get("adjustment_method").and_then(Value::as_str).unwrap_or("");
            if matches!(role, "pre_treatment_confounder" | "effect_modifier") && method == "none" {
                findings.push(format!("{path}/adjustment_method"), "INCOMPLETE", "pre-treatment confounder/effect modifier has no declared adjustment or stratification method");
            }
            if matches!(role, "mediator" | "post_treatment_variable") {
                if protocol.pointer("/estimand/type").and_then(Value::as_str) == Some("average_treatment_effect")
                    && method != "not_adjusted_for_total_effect"
                {
                    findings.push(
                        format!("{path}/adjustment_method"),
                        "INVALID",
                        "total-effect estimand must not adjust for a post-treatment mediator; define a controlled direct effect separately",
                    );
                }
            }
            if let Some(source_id) = confounder.get("source_id").and_then(Value::as_str) {
                if !source_ids.contains(source_id) {
                    findings.push(format!("{path}/source_id"), "INVALID", format!("confounder source '{source_id}' does not resolve in evidence_source_ids"));
                }
            }
        }
    }

    let design = protocol.get("design").unwrap_or(&Value::Null);
    let design_type = design.get("design_type").and_then(Value::as_str).unwrap_or("");
    if design_type == "randomized_switchback" {
        if design.get("randomization_authorized").and_then(Value::as_bool) != Some(true) {
            findings.push("/design/randomization_authorized", "BLOCKED", "randomized switchback requires separate documented owner/master/SMS and applicable class/flag authorization");
        }
        if design.get("authorization_evidence_ids").and_then(Value::as_array).is_none_or(Vec::is_empty) {
            findings.push("/design/authorization_evidence_ids", "BLOCKED", "randomized switchback has no authorization evidence references");
        }
        if design.get("washout_seconds").and_then(Value::as_u64).is_none_or(|v| v == 0) {
            findings.push("/design/washout_seconds", "INCOMPLETE", "switchback design requires a justified positive washout interval");
        }
    }

    if design_type == "matched_observational" || design_type == "target_trial_emulation" {
        let diagnostics = protocol.get("diagnostics").unwrap_or(&Value::Null);
        let min_p = diagnostics.get("minimum_propensity").and_then(Value::as_f64);
        let max_p = diagnostics.get("maximum_propensity").and_then(Value::as_f64);
        match (min_p, max_p) {
            (Some(minimum), Some(maximum)) if minimum >= 0.0 && maximum <= 1.0 && minimum < maximum => {}
            _ => findings.push("/diagnostics", "INVALID", "observational design requires a valid declared propensity/overlap range with 0 <= minimum < maximum <= 1"),
        }
        if diagnostics.get("effective_sample_size").and_then(Value::as_f64).is_none_or(|n| n <= 0.0) {
            findings.push("/diagnostics/effective_sample_size", "INCOMPLETE", "effective sample size is absent or non-positive; overlap/positivity cannot be established");
        }
        if diagnostics.get("maximum_weight").and_then(Value::as_f64).is_none_or(|w| w <= 0.0) {
            findings.push("/diagnostics/maximum_weight", "INCOMPLETE", "weight diagnostics are absent or non-positive");
        }
    }

    let battery = protocol.get("battery_energy_accounting").unwrap_or(&Value::Null);
    let battery_present = battery.get("battery_present").and_then(Value::as_bool);
    let soc_treatment = battery.get("soc_boundary_treatment").and_then(Value::as_str).unwrap_or("");
    match battery_present {
        Some(true) if matches!(soc_treatment, "matched" | "energy_adjusted") => {}
        Some(true) => findings.push(
            "/battery_energy_accounting/soc_boundary_treatment",
            "INCOMPLETE",
            "hybrid comparison must match initial/final SOC or explicitly adjust battery energy and conversion losses",
        ),
        Some(false) if soc_treatment == "not_applicable" => {}
        Some(false) => findings.push(
            "/battery_energy_accounting/soc_boundary_treatment",
            "INVALID",
            "battery is absent but SOC boundary treatment is not not_applicable",
        ),
        None => findings.push("/battery_energy_accounting/battery_present", "INVALID", "battery_present must be explicit"),
    }

    let gates = protocol.get("identification_gates").unwrap_or(&Value::Null);
    if let Some(gates) = gates.as_object() {
        for (name, gate) in gates {
            match gate.get("status").and_then(Value::as_str) {
                Some("pass") | Some("not_applicable") => {}
                Some("fail") => findings.push(
                    format!("/identification_gates/{name}"),
                    "BLOCKED",
                    format!("identification gate '{name}' failed"),
                ),
                Some("unknown") | None => findings.push(
                    format!("/identification_gates/{name}"),
                    "INCOMPLETE",
                    format!("identification gate '{name}' is not established"),
                ),
                _ => findings.push(
                    format!("/identification_gates/{name}"),
                    "INVALID",
                    format!("identification gate '{name}' has an invalid status"),
                ),
            }
        }
    }

    if let Some(controls) = protocol.get("negative_controls").and_then(Value::as_array) {
        let mut ids = BTreeSet::new();
        for (index, control) in controls.iter().enumerate() {
            let path = format!("/negative_controls/{index}");
            if let Some(id) = control.get("control_id").and_then(Value::as_str) {
                if !ids.insert(id.to_owned()) {
                    findings.push(format!("{path}/control_id"), "INVALID", format!("duplicate negative-control id '{id}'"));
                }
            }
            match control.get("status").and_then(Value::as_str) {
                Some("pass") => {}
                Some("fail") => findings.push(path, "BLOCKED", "negative control failed; causal claim is blocked"),
                Some("unknown") | None => findings.push(path, "INCOMPLETE", "negative control has not been run or reviewed"),
                _ => findings.push(path, "INVALID", "unsupported negative-control status"),
            }
        }
    }

    if let Some(evidence_ids) = protocol.get("evidence_source_ids").and_then(Value::as_array) {
        if evidence_ids.is_empty() {
            findings.push("/evidence_source_ids", "INCOMPLETE", "no evidence sources are declared");
        }
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-causal-protocol-audit <schema.json> <protocol.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let protocol = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &protocol, &mut findings);
    check_protocol(&protocol, &mut findings);

    let report = AuditReport {
        auditor: "marine-causal-protocol-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        protocol_id: protocol.get("protocol_id").and_then(Value::as_str).map(str::to_owned),
        readiness_disposition: findings.disposition(),
        claim_ceiling: "Protocol-readiness and declared-identifiability checks only. This does not estimate a causal effect, establish exchangeability, certify an intervention as safe or prove fuel savings.",
        diagnostics: findings.diagnostics,
    };
    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.readiness_disposition == "READY_FOR_INDEPENDENT_REVIEW" { 0 } else { 2 })
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

    const CAUSAL_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-causal-study-protocol-v0.1.0.schema.json");
    const CAUSAL_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-causal-study-protocol-v0.1.0.json");

    fn fixture() -> Value {
        serde_json::from_str(CAUSAL_FIXTURE).expect("causal fixture JSON")
    }

    #[test]
    fn committed_causal_protocol_fixture_matches_schema() {
        let schema: Value = serde_json::from_str(CAUSAL_SCHEMA).expect("causal schema JSON");
        let protocol = fixture();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("causal schema compiles");
        assert!(validator.is_valid(&protocol));
    }

    #[test]
    fn synthetic_protocol_is_not_causal_evidence() {
        let mut findings = Findings::default();
        check_protocol(&fixture(), &mut findings);
        assert_eq!(findings.disposition(), "NOT_READY");
        assert!(findings.incomplete);
    }

    #[test]
    fn total_effect_must_not_adjust_for_treatment_mediator() {
        let mut protocol = fixture();
        protocol["confounders"][0]["causal_role"] = serde_json::json!("mediator");
        protocol["confounders"][0]["adjustment_method"] = serde_json::json!("regression covariate");
        let mut findings = Findings::default();
        check_protocol(&protocol, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn hybrid_treatment_without_soc_closure_is_not_ready() {
        let mut protocol = fixture();
        protocol["battery_energy_accounting"]["soc_boundary_treatment"] = serde_json::json!("unknown");
        let mut findings = Findings::default();
        check_protocol(&protocol, &mut findings);
        assert!(findings.incomplete);
    }

    #[test]
    fn matched_observational_study_without_effective_sample_is_not_identifiable() {
        let mut protocol = fixture();
        protocol["diagnostics"]["effective_sample_size"] = serde_json::json!(0);
        let mut findings = Findings::default();
        check_protocol(&protocol, &mut findings);
        assert!(findings.incomplete);
    }

    #[test]
    fn unauthorized_randomized_switchback_is_blocked() {
        let mut protocol = fixture();
        protocol["design"]["design_type"] = serde_json::json!("randomized_switchback");
        protocol["design"]["randomization_authorized"] = serde_json::json!(false);
        let mut findings = Findings::default();
        check_protocol(&protocol, &mut findings);
        assert!(findings.blocked);
    }

    #[test]
    fn failed_negative_control_blocks_causal_claim() {
        let mut protocol = fixture();
        protocol["negative_controls"][0]["status"] = serde_json::json!("fail");
        let mut findings = Findings::default();
        check_protocol(&protocol, &mut findings);
        assert!(findings.blocked);
    }
}
