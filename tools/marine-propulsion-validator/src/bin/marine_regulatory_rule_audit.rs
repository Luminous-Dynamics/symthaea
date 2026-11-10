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

fn check_v0_2_semantics(register: &Value, findings: &mut Findings) {
    if register.get("schema_version").and_then(Value::as_str) != Some("0.2.0") {
        findings.push("/schema_version", "INVALID", "v0.2 audit requires schema_version 0.2.0");
        return;
    }
    let Some(rules) = register.get("rules").and_then(Value::as_array) else { return; };

    let Some(ets) = rules.iter().find(|rule| rule.get("rule_id").and_then(Value::as_str) == Some("EU-ETS-MARITIME")) else {
        findings.push("/rules", "INCOMPLETE", "v0.2 register has no EU-ETS-MARITIME rule");
        return;
    };
    let calc = ets.get("calculation_contract").unwrap_or(&Value::Null);

    let phases = calc.get("surrender_phase_in").and_then(Value::as_array);
    let Some(phases) = phases else {
        findings.push("/rules/EU-ETS-MARITIME/calculation_contract/surrender_phase_in", "INVALID", "EU ETS v0.2 contract requires explicit emissions-year/surrender-year phase-in");
        return;
    };
    if phases.len() != 3 {
        findings.push(
            "/rules/EU-ETS-MARITIME/calculation_contract/surrender_phase_in",
            "INVALID",
            "current EU ETS phase-in snapshot must contain the 2024, 2025 and 2026 emissions-year transitions",
        );
    }

    let expected_phase_in = [
        (2024_i64, 2025_i64, 0.4_f64),
        (2025_i64, 2026_i64, 0.7_f64),
        (2026_i64, 2027_i64, 1.0_f64),
    ];
    for (index, expected) in expected_phase_in.iter().enumerate() {
        let observed = phases.get(index).and_then(|phase| {
            Some((
                phase.get("emissions_year")?.as_i64()?,
                phase.get("surrender_year")?.as_i64()?,
                phase.get("surrender_fraction")?.as_f64()?,
            ))
        });
        if observed.is_some_and(|value| value != *expected) {
            findings.push(
                format!("/rules/EU-ETS-MARITIME/calculation_contract/surrender_phase_in/{index}"),
                "INVALID",
                format!("EU ETS phase-in does not match expected emissions/surrender years and fraction: {:?}", expected),
            );
        }
    }

    let mut emissions_years = BTreeSet::new();
    let mut surrender_years = BTreeSet::new();
    let mut previous_emissions_year: Option<i64> = None;
    let mut previous_fraction: Option<f64> = None;

    for (index, phase) in phases.iter().enumerate() {
        let path = format!("/rules/EU-ETS-MARITIME/calculation_contract/surrender_phase_in/{index}");
        let emissions_year = phase.get("emissions_year").and_then(Value::as_i64);
        let surrender_year = phase.get("surrender_year").and_then(Value::as_i64);
        let fraction = phase.get("surrender_fraction").and_then(Value::as_f64);

        match (emissions_year, surrender_year) {
            (Some(emissions), Some(surrender)) if surrender > emissions => {}
            (Some(_), Some(_)) => findings.push(
                path.clone(),
                "INVALID",
                "surrender_year must be later than emissions_year",
            ),
            _ => findings.push(
                path.clone(),
                "INVALID",
                "phase-in must declare integer emissions_year and surrender_year",
            ),
        }

        if let Some(year) = emissions_year {
            if !emissions_years.insert(year) {
                findings.push(format!("{path}/emissions_year"), "INVALID", "duplicate emissions_year in phase-in schedule");
            }
            if previous_emissions_year.is_some_and(|previous| year <= previous) {
                findings.push(format!("{path}/emissions_year"), "INVALID", "phase-in entries must be ordered by strictly increasing emissions_year");
            }
            previous_emissions_year = Some(year);
        }
        if let Some(year) = surrender_year {
            if !surrender_years.insert(year) {
                findings.push(format!("{path}/surrender_year"), "INVALID", "duplicate surrender_year in phase-in schedule");
            }
        }
        if let Some(fraction) = fraction {
            if !(0.0..=1.0).contains(&fraction) {
                findings.push(format!("{path}/surrender_fraction"), "INVALID", "surrender_fraction must be in [0,1]");
            }
            if previous_fraction.is_some_and(|previous| fraction < previous) {
                findings.push(format!("{path}/surrender_fraction"), "INVALID", "surrender fractions cannot decrease across later emissions years in the declared phase-in schedule");
            }
            previous_fraction = Some(fraction);
        } else {
            findings.push(format!("{path}/surrender_fraction"), "INVALID", "surrender_fraction must be a numeric fraction in [0,1]");
        }
    }

    let coverage = calc.get("voyage_coverage").and_then(Value::as_array);
    if let Some(coverage) = coverage {
        let mut coverage_ids = BTreeSet::new();
        for (index, item) in coverage.iter().enumerate() {
            let path = format!("/rules/EU-ETS-MARITIME/calculation_contract/voyage_coverage/{index}");
            if let Some(id) = item.get("coverage_id").and_then(Value::as_str) {
                if !coverage_ids.insert(id.to_owned()) {
                    findings.push(format!("{path}/coverage_id"), "INVALID", format!("duplicate voyage coverage id '{id}'"));
                }
            }
            if item.get("covered_fraction").and_then(Value::as_f64).is_none_or(|fraction| !(0.0..=1.0).contains(&fraction)) {
                findings.push(format!("{path}/covered_fraction"), "INVALID", "covered_fraction must be a numeric fraction in [0,1]");
            }
        }
        for required in ["EU_TO_EU", "EU_TO_NON_EU", "NON_EU_TO_EU", "WITHIN_EU_PORT"] {
            if !coverage_ids.contains(required) {
                findings.push(
                    "/rules/EU-ETS-MARITIME/calculation_contract/voyage_coverage",
                    "INCOMPLETE",
                    format!("required voyage coverage case '{required}' is not represented"),
                );
            }
        }
        let expected_coverage = [
            ("EU_TO_EU", 1.0_f64),
            ("EU_TO_NON_EU", 0.5_f64),
            ("NON_EU_TO_EU", 0.5_f64),
            ("WITHIN_EU_PORT", 1.0_f64),
        ];
        for (coverage_id, expected_fraction) in expected_coverage {
            let observed = coverage.iter().find(|item| {
                item.get("coverage_id").and_then(Value::as_str) == Some(coverage_id)
            }).and_then(|item| item.get("covered_fraction")).and_then(Value::as_f64);
            if observed.is_some_and(|fraction| (fraction - expected_fraction).abs() > f64::EPSILON) {
                findings.push(
                    "/rules/EU-ETS-MARITIME/calculation_contract/voyage_coverage",
                    "INVALID",
                    format!("coverage fraction for {coverage_id} must be {expected_fraction:.1} under this rule snapshot"),
                );
            }
        }
    } else {
        findings.push(
            "/rules/EU-ETS-MARITIME/calculation_contract/voyage_coverage",
            "INCOMPLETE",
            "v0.2 route-specific ETS coverage fractions are absent",
        );
    }

    let gases = calc.get("gas_scope").and_then(Value::as_array);
    if let Some(gases) = gases {
        let mut gas_years = std::collections::BTreeMap::new();
        for (index, item) in gases.iter().enumerate() {
            let path = format!("/rules/EU-ETS-MARITIME/calculation_contract/gas_scope/{index}");
            let gas = item.get("gas").and_then(Value::as_str);
            let year = item.get("included_from_emissions_year").and_then(Value::as_i64);
            if let (Some(gas), Some(year)) = (gas, year) {
                if gas_years.insert(gas.to_owned(), year).is_some() {
                    findings.push(format!("{path}/gas"), "INVALID", format!("duplicate gas scope entry '{gas}'"));
                }
            } else {
                findings.push(path, "INVALID", "gas scope requires gas and included_from_emissions_year");
            }
        }
        if gas_years.get("CO2") != Some(&2024)
            || gas_years.get("CH4") != Some(&2026)
            || gas_years.get("N2O") != Some(&2026)
        {
            findings.push(
                "/rules/EU-ETS-MARITIME/calculation_contract/gas_scope",
                "INVALID",
                "current EU ETS gas scope must retain CO2 from 2024 and CH4/N2O from 2026",
            );
        }
    } else {
        findings.push(
            "/rules/EU-ETS-MARITIME/calculation_contract/gas_scope",
            "INCOMPLETE",
            "gas-specific emissions start years are absent",
        );
    }

    if let Some(mrv) = rules.iter().find(|rule| rule.get("rule_id").and_then(Value::as_str) == Some("EU-MRV-MARITIME")) {
        let gases = mrv.pointer("/calculation_contract/gas_scope").and_then(Value::as_array);
        let mut observed = std::collections::BTreeMap::new();
        if let Some(gases) = gases {
            for gas in gases {
                if let (Some(name), Some(year)) = (
                    gas.get("gas").and_then(Value::as_str),
                    gas.get("included_from_emissions_year").and_then(Value::as_i64),
                ) {
                    observed.insert(name.to_owned(), year);
                }
            }
        }
        if observed.get("CO2") != Some(&2018)
            || observed.get("CH4") != Some(&2024)
            || observed.get("N2O") != Some(&2024)
        {
            findings.push(
                "/rules/EU-MRV-MARITIME/calculation_contract/gas_scope",
                "INVALID",
                "EU MRV gas scope must keep CO2 from 2018 and CH4/N2O from 2024 in this snapshot, separately from ETS inclusion years",
            );
        }
    } else {
        findings.push("/rules", "INCOMPLETE", "v0.2 register has no EU-MRV-MARITIME rule separate from EU ETS");
    }

    if let Some(eca) = rules.iter().find(|rule| rule.get("rule_id").and_then(Value::as_str) == Some("NE-ATLANTIC-ECA-2026")) {
        let dates = eca.get("dates").unwrap_or(&Value::Null);
        if dates.get("entry_into_force_on").and_then(Value::as_str) != Some("2027-09-01")
            || dates.get("applies_from").and_then(Value::as_str) != Some("2028-09-01")
        {
            findings.push(
                "/rules/NE-ATLANTIC-ECA-2026/dates",
                "INVALID",
                "North-East Atlantic ECA snapshot must distinguish 2027-09-01 entry into force from 2028-09-01 application",
            );
        }
    } else {
        findings.push("/rules", "INCOMPLETE", "v0.2 register has no North-East Atlantic ECA rule");
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
    if register.get("schema_version").and_then(Value::as_str) == Some("0.2.0") {
        check_v0_2_semantics(&register, &mut findings);
    }

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
    const RULE_SCHEMA_V2: &str =
        include_str!("../../../../docs/research/marine/marine-regulatory-rule-register-v0.2.0.schema.json");
    const RULE_REGISTER_V2: &str =
        include_str!("../../../../docs/research/marine/marine-regulatory-rule-register-2026-11-10-v0.2.0.json");

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
    fn v0_2_regulatory_register_matches_schema_and_explicit_timing_model() {
        let schema: Value = serde_json::from_str(RULE_SCHEMA_V2).expect("v0.2 rule schema JSON");
        let register: Value = serde_json::from_str(RULE_REGISTER_V2).expect("v0.2 rule register JSON");
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("v0.2 schema compiles");
        assert!(validator.is_valid(&register));
        let mut findings = Findings::default();
        check_registry_semantics(&register, &mut findings);
        check_v0_2_semantics(&register, &mut findings);
        assert!(!findings.invalid, "{:?}", findings.diagnostics);
    }

    #[test]
    fn v0_2_ets_surrender_year_must_follow_emissions_year() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER_V2).expect("v0.2 rule register JSON");
        register["rules"][2]["calculation_contract"]["surrender_phase_in"][0]["surrender_year"] =
            serde_json::json!(2024);
        let mut findings = Findings::default();
        check_v0_2_semantics(&register, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_2_ets_coverage_is_not_surrender_phase_in() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER_V2).expect("v0.2 rule register JSON");
        register["rules"][2]["calculation_contract"]["voyage_coverage"][1]["covered_fraction"] =
            serde_json::json!(0.7);
        let mut findings = Findings::default();
        check_v0_2_semantics(&register, &mut findings);
        assert!(!findings.invalid);
        assert!(!findings.incomplete);
        // Coverage fraction (50%) is a different axis from the surrender phase-in (70%).
        assert_ne!(
            register["rules"][2]["calculation_contract"]["voyage_coverage"][1]["covered_fraction"],
            register["rules"][2]["calculation_contract"]["surrender_phase_in"][1]["surrender_fraction"]
        );
    }

    #[test]
    fn v0_2_mrv_and_ets_gas_years_are_not_conflated() {
        let register: Value = serde_json::from_str(RULE_REGISTER_V2).expect("v0.2 rule register JSON");
        let ets = register["rules"].as_array().unwrap().iter().find(|rule|
            rule.get("rule_id").and_then(Value::as_str) == Some("EU-ETS-MARITIME")
        ).unwrap();
        let mrv = register["rules"].as_array().unwrap().iter().find(|rule|
            rule.get("rule_id").and_then(Value::as_str) == Some("EU-MRV-MARITIME")
        ).unwrap();
        assert_eq!(ets.pointer("/calculation_contract/gas_scope/1/included_from_emissions_year").and_then(Value::as_i64), Some(2026));
        assert_eq!(mrv.pointer("/calculation_contract/gas_scope/1/included_from_emissions_year").and_then(Value::as_i64), Some(2024));
    }

    #[test]
    fn v0_2_ets_coverage_fraction_cannot_be_confused_with_surrender_fraction() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER_V2).expect("v0.2 rule register JSON");
        let ets = register["rules"].as_array_mut().unwrap().iter_mut().find(|rule|
            rule.get("rule_id").and_then(Value::as_str) == Some("EU-ETS-MARITIME")
        ).unwrap();
        ets["calculation_contract"]["voyage_coverage"][1]["covered_fraction"] = serde_json::json!(0.7);
        let mut findings = Findings::default();
        check_v0_2_semantics(&register, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn v0_2_north_east_atlantic_eca_keeps_entry_and_application_dates_distinct() {
        let mut register: Value = serde_json::from_str(RULE_REGISTER_V2).expect("v0.2 rule register JSON");
        let eca_index = register["rules"].as_array().unwrap().iter().position(|rule|
            rule.get("rule_id").and_then(Value::as_str) == Some("NE-ATLANTIC-ECA-2026")
        ).unwrap();
        register["rules"][eca_index]["dates"]["entry_into_force_on"] = serde_json::json!("2028-09-01");
        let mut findings = Findings::default();
        check_v0_2_semantics(&register, &mut findings);
        assert!(findings.invalid);
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
