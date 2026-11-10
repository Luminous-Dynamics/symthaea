use chrono::NaiveDate;
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
    registry_id: Option<String>,
    snapshot_date: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    rule_count: usize,
    source_count: usize,
    diagnostics: Vec<Diagnostic>,
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
            "REGISTER_CONTRACT_VALID"
        }
    }
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn valid_date(value: Option<&str>) -> Option<NaiveDate> {
    value.and_then(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
}

fn validate_schema(schema: &Value, register: &Value, findings: &mut Findings) {
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
    for error in validator.iter_errors(register) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("registry schema violation: {error}"),
        );
    }
}

fn check_registry_semantics(register: &Value, findings: &mut Findings) {
    let snapshot = register.get("snapshot_date").and_then(Value::as_str);
    let snapshot_date = valid_date(snapshot);
    if snapshot_date.is_none() {
        findings.push("/snapshot_date", "INVALID", "snapshot_date must be a valid YYYY-MM-DD date");
    }

    let sources = register.get("sources").and_then(Value::as_array);
    let rules = register.get("rules").and_then(Value::as_array);
    let Some(sources) = sources else { return };
    let Some(rules) = rules else { return };

    let mut source_ids = BTreeSet::new();
    for (index, source) in sources.iter().enumerate() {
        let path = format!("/sources/{index}/source_id");
        match source.get("source_id").and_then(Value::as_str) {
            Some(id) if !source_ids.insert(id.to_owned()) => {
                findings.push(path, "INVALID", format!("duplicate regulatory source id '{id}'"));
            }
            Some(_) => {}
            None => findings.push(path, "INVALID", "source_id is missing"),
        }
    }

    let mut rule_ids = BTreeSet::new();
    for (index, rule) in rules.iter().enumerate() {
        let path = format!("/rules/{index}");
        let id = rule.get("rule_id").and_then(Value::as_str);
        if let Some(id) = id {
            if !rule_ids.insert(id.to_owned()) {
                findings.push(format!("{path}/rule_id"), "INVALID", format!("duplicate rule id '{id}'"));
            }
        }

        let status = rule.get("rule_status").and_then(Value::as_str).unwrap_or("");
        let legal_character = rule.get("legal_character").and_then(Value::as_str).unwrap_or("");
        let dates = rule.get("dates").unwrap_or(&Value::Null);

        let adopted = dates.get("adopted_on").and_then(Value::as_str);
        let entry_into_force = dates.get("entry_into_force_on").and_then(Value::as_str);
        let applies_from = dates.get("applies_from").and_then(Value::as_str);
        let applies_from_year = dates.get("applies_from_year").and_then(Value::as_i64);
        let scheduled_from = dates.get("review_scheduled_from").and_then(Value::as_str);
        let scheduled_through = dates.get("review_scheduled_through").and_then(Value::as_str);

        for (field, value) in [
            ("adopted_on", adopted),
            ("entry_into_force_on", entry_into_force),
            ("applies_from", applies_from),
            ("review_scheduled_from", scheduled_from),
            ("review_scheduled_through", scheduled_through),
        ] {
            if value.is_some() && valid_date(value).is_none() {
                findings.push(format!("{path}/dates/{field}"), "INVALID", "date must be a valid YYYY-MM-DD value");
            }
        }

        if let (Some(entry), Some(application)) = (valid_date(entry_into_force), valid_date(applies_from)) {
            if application < entry {
                findings.push(
                    format!("{path}/dates/applies_from"),
                    "INVALID",
                    "application date cannot precede entry into force",
                );
            }
        }

        match status {
            "proposed" | "under_review" => {
                if entry_into_force.is_some() || applies_from.is_some() || applies_from_year.is_some() {
                    findings.push(
                        format!("{path}/dates"),
                        "INVALID",
                        "proposed/under-review rules cannot carry an effective application date",
                    );
                }
                if adopted.is_some() {
                    findings.push(
                        format!("{path}/dates/adopted_on"),
                        "INVALID",
                        "proposed/under-review status conflicts with adopted_on; resolve status before use",
                    );
                }
                if scheduled_from.is_none() && status == "under_review" {
                    findings.push(
                        format!("{path}/dates/review_scheduled_from"),
                        "INCOMPLETE",
                        "under-review rule has no dated review snapshot or scheduled review marker",
                    );
                }
            }
            "adopted" => {
                if valid_date(adopted).is_none() {
                    findings.push(format!("{path}/dates/adopted_on"), "INVALID", "adopted rule must declare adopted_on");
                }
                if applies_from.is_none() && applies_from_year.is_none() {
                    findings.push(
                        format!("{path}/dates"),
                        "INCOMPLETE",
                        "adopted measure should declare its application date/year when known",
                    );
                }
            }
            "guidance_adopted" => {
                if valid_date(adopted).is_none() {
                    findings.push(format!("{path}/dates/adopted_on"), "INVALID", "adopted guidance must declare adopted_on");
                }
            }
            "in_force" | "applicable" => {
                if entry_into_force.is_none() && applies_from.is_none() && applies_from_year.is_none() {
                    findings.push(
                        format!("{path}/dates"),
                        "INVALID",
                        "in-force/applicable rule must declare an entry-into-force or application date",
                    );
                }
            }
            "superseded" | "withdrawn" => {}
            _ => findings.push(format!("{path}/rule_status"), "INVALID", "unsupported rule status"),
        }

        if let (Some(snapshot), Some(publication)) = (
            snapshot_date,
            valid_date(rule.get("dates").and_then(|v| v.get("adopted_on")).and_then(Value::as_str)),
        ) {
            if publication > snapshot {
                findings.push(
                    format!("{path}/dates/adopted_on"),
                    "INVALID",
                    "adoption date is later than the registry snapshot date",
                );
            }
        }

        if legal_character == "negotiation_proposal" && status != "proposed" && status != "under_review" {
            findings.push(
                format!("{path}/legal_character"),
                "INVALID",
                "negotiation proposal cannot be marked adopted/in-force without a new authoritative record",
            );
        }

        let rule_sources = rule.get("source_ids").and_then(Value::as_array);
        if let Some(rule_sources) = rule_sources {
            for (source_index, source) in rule_sources.iter().enumerate() {
                if let Some(source_id) = source.as_str() {
                    if !source_ids.contains(source_id) {
                        findings.push(
                            format!("{path}/source_ids/{source_index}"),
                            "INVALID",
                            format!("source id '{source_id}' does not resolve in /sources"),
                        );
                    }
                }
            }
        }
    }
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-regulatory-rule-audit <schema.json> <register.json>".to_owned());
    }

    let schema = read_json(&args[1])?;
    let register = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &register, &mut findings);
    check_registry_semantics(&register, &mut findings);

    let report = AuditReport {
        auditor: "marine-regulatory-rule-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        registry_id: register.get("registry_id").and_then(Value::as_str).map(str::to_owned),
        snapshot_date: register.get("snapshot_date").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Registry/schema/date/status/reference checks only. This is not a legal applicability decision, verifier statement or regulatory compliance approval.",
        rule_count: register.get("rules").and_then(Value::as_array).map_or(0, Vec::len),
        source_count: register.get("sources").and_then(Value::as_array).map_or(0, Vec::len),
        diagnostics: findings.diagnostics,
    };

    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "REGISTER_CONTRACT_VALID" { 0 } else { 2 })
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

    const RULE_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-regulatory-rule-register-v0.1.0.schema.json");
    const RULE_REGISTER: &str =
        include_str!("../../../../docs/research/marine/marine-regulatory-rule-register-2026-11-10.json");

    #[test]
    fn committed_regulatory_rule_register_matches_its_schema() {
        let schema: Value = serde_json::from_str(RULE_SCHEMA).expect("rule schema JSON");
        let register: Value = serde_json::from_str(RULE_REGISTER).expect("rule register JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("schema compiles");
        assert!(validator.is_valid(&register));
    }

    #[test]
    fn all_rule_source_references_resolve() {
        let register: Value = serde_json::from_str(RULE_REGISTER).expect("rule register JSON");
        let mut findings = Findings::default();
        check_registry_semantics(&register, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn pending_review_cannot_claim_effective_date() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER).expect("rule register JSON");
        register["rules"][0]["dates"]["applies_from"] = serde_json::json!("2028-01-01");
        let mut findings = Findings::default();
        check_registry_semantics(&register, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn binding_rule_requires_application_date() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER).expect("rule register JSON");
        register["rules"][2]["dates"].as_object_mut().expect("date object").remove("applies_from");
        let mut findings = Findings::default();
        check_registry_semantics(&register, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn negotiation_proposal_cannot_be_marked_in_force() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER).expect("rule register JSON");
        register["rules"][5]["rule_status"] = serde_json::json!("in_force");
        register["rules"][5]["dates"]["applies_from"] = serde_json::json!("2027-01-01");
        let mut findings = Findings::default();
        check_registry_semantics(&register, &mut findings);
        assert!(findings.invalid);
    }
}
