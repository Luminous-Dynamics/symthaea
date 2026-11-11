use chrono::{DateTime, Utc};
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
            "BIOFOULING_EVIDENCE_CONTRACT_VALID"
        }
    }
}

#[derive(Debug, Serialize)]
struct AuditReport {
    auditor: &'static str,
    auditor_version: &'static str,
    record_id: Option<String>,
    disposition: &'static str,
    claim_ceiling: &'static str,
    hull_coverage_fraction: Option<f64>,
    zone_count: usize,
    confirmed_species_count: usize,
    cleaning_method: Option<String>,
    performance_comparison_eligible: bool,
    diagnostics: Vec<Diagnostic>,
}

fn read_json(path: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid JSON in {path}: {error}"))
}

fn parse_time(value: Option<&str>) -> Option<DateTime<Utc>> {
    value
        .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
        .map(|date| date.with_timezone(&Utc))
}

fn validate_schema(schema: &Value, record: &Value, findings: &mut Findings) {
    if schema.get("$schema").and_then(Value::as_str) != Some(DRAFT_2020_12) {
        findings.push("/$schema", "INVALID", "only JSON Schema Draft 2020-12 is supported");
        return;
    }
    if !jsonschema::draft202012::meta::is_valid(schema) {
        findings.push("/", "INVALID", "biofouling schema does not validate against the Draft 2020-12 meta-schema");
        return;
    }
    let validator = match jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(schema)
    {
        Ok(validator) => validator,
        Err(error) => {
            findings.push("/", "INVALID", format!("cannot compile biofouling schema: {error}"));
            return;
        }
    };
    for error in validator.iter_errors(record) {
        findings.push(
            error.instance_path().to_string(),
            "INVALID",
            format!("biofouling record schema violation: {error}"),
        );
    }
}

fn check_species(zone: &Value, index: usize, source_ids: &BTreeSet<String>, findings: &mut Findings) {
    let path = format!("/inspection/zones/{index}");
    match zone.get("species_status").and_then(Value::as_str) {
        Some("confirmed") => {
            let taxon = zone.get("taxonomic_identification").and_then(Value::as_str);
            let source = zone.get("species_evidence_source_id").and_then(Value::as_str);
            if taxon.is_none_or(|value| value.trim().is_empty()) {
                findings.push(
                    format!("{path}/taxonomic_identification"),
                    "INVALID",
                    "confirmed species identification requires an explicit taxonomic identification",
                );
            }
            match source {
                Some(source_id) if source_ids.contains(source_id) => {}
                Some(source_id) => findings.push(
                    format!("{path}/species_evidence_source_id"),
                    "INVALID",
                    format!("species evidence source '{source_id}' does not resolve in /evidence_sources"),
                ),
                None => findings.push(
                    format!("{path}/species_evidence_source_id"),
                    "INVALID",
                    "confirmed species identification requires a source record",
                ),
            }
        }
        Some("suspected") => findings.push(
            format!("{path}/species_status"),
            "INCOMPLETE",
            "suspected species requires taxonomic follow-up; do not promote suspicion to confirmed identity",
        ),
        Some("unknown") | None => findings.push(
            format!("{path}/species_status"),
            "INCOMPLETE",
            "species status is unknown; absence of identification is not evidence of no invasive-species risk",
        ),
        Some("not_assessed") | Some("none_observed") => {}
        _ => findings.push(
            format!("{path}/species_status"),
            "INVALID",
            "unsupported species status",
        ),
    }
}

fn check_inspection(record: &Value, findings: &mut Findings) -> (Option<f64>, usize, usize) {
    let inspection = record.get("inspection").unwrap_or(&Value::Null);
    let total_area = inspection.get("total_hull_area_m2").and_then(Value::as_f64);
    let inspected_area = inspection.get("inspected_area_m2").and_then(Value::as_f64);
    let declared_fraction = inspection.get("declared_total_coverage_fraction").and_then(Value::as_f64);
    let minimum_fraction = inspection.get("minimum_required_coverage_fraction").and_then(Value::as_f64);
    let zones = inspection.get("zones").and_then(Value::as_array);
    let Some(zones) = zones else {
        findings.push("/inspection/zones", "INVALID", "inspection must contain at least one hull zone");
        return (None, 0, 0);
    };

    let mut ids = BTreeSet::new();
    let mut zone_area_sum = 0.0;
    let mut zone_inspected_sum = 0.0;
    let mut confirmed_species_count = 0;

    let source_ids: BTreeSet<String> = record
        .get("evidence_sources")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|source| source.get("source_id").and_then(Value::as_str).map(str::to_owned))
        .collect();

    for (index, zone) in zones.iter().enumerate() {
        let path = format!("/inspection/zones/{index}");
        match zone.get("zone_id").and_then(Value::as_str) {
            Some(id) if !ids.insert(id.to_owned()) => findings.push(
                format!("{path}/zone_id"),
                "INVALID",
                format!("duplicate hull-zone identity '{id}'"),
            ),
            Some(_) => {}
            None => findings.push(format!("{path}/zone_id"), "INVALID", "zone identity is missing"),
        }

        let zone_area = zone.get("zone_area_m2").and_then(Value::as_f64);
        let inspected = zone.get("inspected_area_m2").and_then(Value::as_f64);
        match (zone_area, inspected) {
            (Some(area), Some(visited)) if area > 0.0 && visited >= 0.0 && visited <= area => {
                zone_area_sum += area;
                zone_inspected_sum += visited;
                let actual = visited / area;
                if let Some(declared) = zone.get("declared_inspection_coverage_fraction").and_then(Value::as_f64) {
                    if (actual - declared).abs() > 0.005 {
                        findings.push(
                            format!("{path}/declared_inspection_coverage_fraction"),
                            "INVALID",
                            format!("declared zone coverage {declared:.4} differs from area-derived coverage {actual:.4}"),
                        );
                    }
                }
            }
            (Some(_), Some(_)) => findings.push(
                &path,
                "INVALID",
                "zone inspected area must be between zero and the positive zone area",
            ),
            _ => findings.push(&path, "INVALID", "zone area and inspected area must be numeric"),
        }

        let fouling_class = zone.get("fouling_class").and_then(Value::as_str);
        let fouling_fraction = zone.get("estimated_fouling_coverage_fraction").and_then(Value::as_f64);
        if fouling_class == Some("none") && fouling_fraction.is_some_and(|value| value > 0.0) {
            findings.push(
                &path,
                "INVALID",
                "fouling_class 'none' contradicts positive estimated fouling coverage",
            );
        }
        if matches!(fouling_class, Some("unknown") | None) {
            findings.push(
                format!("{path}/fouling_class"),
                "INCOMPLETE",
                "fouling class is unknown; preserve the uncertainty",
            );
        }

        if zone.get("species_status").and_then(Value::as_str) == Some("confirmed") {
            confirmed_species_count += 1;
        }
        check_species(zone, index, &source_ids, findings);
    }

    let Some(total_area) = total_area else {
        findings.push("/inspection/total_hull_area_m2", "INVALID", "total hull area is missing");
        return (None, zones.len(), confirmed_species_count);
    };
    let Some(inspected_area) = inspected_area else {
        findings.push("/inspection/inspected_area_m2", "INVALID", "inspected hull area is missing");
        return (None, zones.len(), confirmed_species_count);
    };
    if total_area <= 0.0 || inspected_area < 0.0 || inspected_area > total_area {
        findings.push(
            "/inspection",
            "INVALID",
            "inspected hull area must be between zero and the positive total hull area",
        );
        return (None, zones.len(), confirmed_species_count);
    }

    if (zone_area_sum - total_area).abs() > (total_area * 0.005).max(0.01) {
        findings.push(
            "/inspection/zones",
            "INCOMPLETE",
            format!(
                "zone-area sum {zone_area_sum:.3} m² does not account for total hull area {total_area:.3} m² within the 0.5% coverage tolerance"
            ),
        );
    }
    if (zone_inspected_sum - inspected_area).abs() > (total_area * 0.005).max(0.01) {
        findings.push(
            "/inspection/inspected_area_m2",
            "INVALID",
            format!(
                "overall inspected area {inspected_area:.3} m² differs from summed zone inspections {zone_inspected_sum:.3} m²"
            ),
        );
    }

    let actual_fraction = inspected_area / total_area;
    if declared_fraction.is_some_and(|value| (actual_fraction - value).abs() > 0.005) {
        findings.push(
            "/inspection/declared_total_coverage_fraction",
            "INVALID",
            format!(
                "declared total coverage differs from area-derived fraction {actual_fraction:.4}"
            ),
        );
    }
    if minimum_fraction.is_some_and(|minimum| actual_fraction < minimum) {
        findings.push(
            "/inspection/inspected_area_m2",
            "INCOMPLETE",
            format!(
                "observed hull coverage {actual_fraction:.4} is below the predeclared minimum {}",
                minimum_fraction.unwrap_or_default()
            ),
        );
    }

    if inspection.get("method").and_then(Value::as_str) == Some("unknown") {
        findings.push("/inspection/method", "INCOMPLETE", "inspection method is unknown");
    }
    if inspection.get("image_or_observation_count").and_then(Value::as_u64).unwrap_or(0) == 0 {
        findings.push(
            "/inspection/image_or_observation_count",
            "INVALID",
            "inspection must retain at least one raw observation",
        );
    }

    (Some(actual_fraction), zones.len(), confirmed_species_count)
}

fn check_cleaning_event(record: &Value, findings: &mut Findings) -> Option<String> {
    let event = record.get("cleaning_event")?;
    if event.is_null() {
        return None;
    }
    let method = event.get("method").and_then(Value::as_str).unwrap_or("");
    let permit = event.get("local_authorization_status").and_then(Value::as_str).unwrap_or("");
    let containment = event.get("containment_method").and_then(Value::as_str).unwrap_or("");
    let waste = event.get("waste_disposition").and_then(Value::as_str).unwrap_or("");

    let start = parse_time(event.get("start").and_then(Value::as_str));
    let end = parse_time(event.get("end").and_then(Value::as_str));
    match (start, end) {
        (Some(start), Some(end)) if start < end => {}
        (Some(_), Some(_)) => findings.push(
            "/cleaning_event/end",
            "INVALID",
            "cleaning event end must be strictly later than start",
        ),
        _ => findings.push(
            "/cleaning_event",
            "INVALID",
            "cleaning event timestamps must be valid RFC3339 date-times",
        ),
    }

    if method == "no_cleaning" {
        if event.get("area_cleaned_m2").and_then(Value::as_f64).is_some_and(|value| value > 0.0) {
            findings.push(
                "/cleaning_event/area_cleaned_m2",
                "INVALID",
                "no_cleaning event cannot declare a positive cleaned area",
            );
        }
        return Some(method.to_owned());
    }

    match permit {
        "approved" => {
            if event.get("authorization_reference").and_then(Value::as_str).is_none_or(|value| value.trim().is_empty()) {
                findings.push(
                    "/cleaning_event/authorization_reference",
                    "INVALID",
                    "approved cleaning requires an authorization reference",
                );
            }
        }
        "not_required" => {
            if event.get("authorization_reason").and_then(Value::as_str).is_none_or(|value| value.trim().is_empty()) {
                findings.push(
                    "/cleaning_event/authorization_reason",
                    "INCOMPLETE",
                    "not_required authorization status needs a jurisdiction/rule reason",
                );
            }
        }
        "pending" | "denied" => findings.push(
            "/cleaning_event/local_authorization_status",
            "BLOCKED",
            "cleaning is not authorized; do not treat the operation as eligible",
        ),
        "unknown" | "" => findings.push(
            "/cleaning_event/local_authorization_status",
            "INCOMPLETE",
            "local cleaning authorization is unknown",
        ),
        _ => findings.push(
            "/cleaning_event/local_authorization_status",
            "INVALID",
            "unsupported local authorization status",
        ),
    }

    if method.starts_with("in_water_") {
        match containment {
            "full_capture" => {
                if !matches!(waste, "contained_disposed" | "filtered_return") {
                    findings.push(
                        "/cleaning_event/waste_disposition",
                        "BLOCKED",
                        "full-capture cleaning requires evidenced contained disposal or filtered return",
                    );
                }
            }
            "partial_capture" | "unknown" | "not_applicable" => findings.push(
                "/cleaning_event/containment_method",
                "INCOMPLETE",
                "in-water cleaning capture/containment is partial, unknown or not applicable",
            ),
            "none" => findings.push(
                "/cleaning_event/containment_method",
                "BLOCKED",
                "in-water cleaning without containment is blocked by this research evidence policy",
            ),
            _ => findings.push(
                "/cleaning_event/containment_method",
                "INVALID",
                "unsupported containment method",
            ),
        }
    }

    match waste {
        "contained_disposed" | "filtered_return" => {}
        "permitted_discharge" => {
            if event.get("waste_authorization_reference").and_then(Value::as_str).is_none_or(|value| value.trim().is_empty()) {
                findings.push(
                    "/cleaning_event/waste_authorization_reference",
                    "BLOCKED",
                    "permitted discharge requires an explicit local waste authorization reference",
                );
            } else {
                findings.push(
                    "/cleaning_event/waste_disposition",
                    "INCOMPLETE",
                    "permitted discharge still requires independent waste-quality and biological-release evidence",
                );
            }
        }
        "uncontrolled_release" => findings.push(
            "/cleaning_event/waste_disposition",
            "BLOCKED",
            "uncontrolled cleaning-waste or organism release is not an eligible operation",
        ),
        "unknown" | "not_applicable" => findings.push(
            "/cleaning_event/waste_disposition",
            "INCOMPLETE",
            "cleaning waste disposition is unknown or not applicable",
        ),
        _ => findings.push(
            "/cleaning_event/waste_disposition",
            "INVALID",
            "unsupported cleaning waste disposition",
        ),
    }

    if event.get("coating_damage_observed").and_then(Value::as_bool) == Some(true) {
        findings.push(
            "/cleaning_event/coating_damage_observed",
            "INCOMPLETE",
            "post-cleaning coating damage requires repair/inspection disposition",
        );
    }

    Some(method.to_owned())
}

fn check_performance_comparison(record: &Value, cleaning_method: Option<&str>, findings: &mut Findings) -> bool {
    let Some(comparison) = record.get("performance_comparison") else {
        findings.push(
            "/performance_comparison",
            "INCOMPLETE",
            "no same-vessel performance comparison supplied; no fuel/shaft-power improvement can be claimed",
        );
        return false;
    };
    if comparison.is_null() {
        findings.push(
            "/performance_comparison",
            "INCOMPLETE",
            "performance comparison is explicitly unavailable",
        );
        return false;
    }

    let before = comparison.get("before").unwrap_or(&Value::Null);
    let after = comparison.get("after").unwrap_or(&Value::Null);
    let before_fingerprint = before.get("condition_fingerprint").and_then(Value::as_str);
    let after_fingerprint = after.get("condition_fingerprint").and_then(Value::as_str);
    if before_fingerprint.is_none() || before_fingerprint != after_fingerprint {
        findings.push(
            "/performance_comparison",
            "INCOMPARABLE",
            "before/after condition fingerprints differ or are missing; no causal performance delta is eligible",
        );
        return false;
    }

    let before_time = parse_time(before.get("observed_at").and_then(Value::as_str));
    let after_time = parse_time(after.get("observed_at").and_then(Value::as_str));
    match (before_time, after_time) {
        (Some(before_time), Some(after_time)) if before_time < after_time => {}
        (Some(_), Some(_)) => findings.push(
            "/performance_comparison/after/observed_at",
            "INVALID",
            "post-intervention observation must be later than baseline observation",
        ),
        _ => findings.push(
            "/performance_comparison",
            "INVALID",
            "performance comparison timestamps must be valid RFC3339 values",
        ),
    }

    for field in [
        "speed_through_water_kn",
        "shaft_rpm",
        "draft_m",
        "trim_m",
        "sea_state",
    ] {
        if before.get(field) != after.get(field) {
            findings.push(
                format!("/performance_comparison/after/{field}"),
                "INCOMPARABLE",
                format!("before/after {field} differs; no reviewed counterfactual profile is supplied"),
            );
        }
    }

    if let Some(method) = cleaning_method {
        if method != "no_cleaning" {
            let cleaning = record.get("cleaning_event").unwrap_or(&Value::Null);
            let cleaning_end = parse_time(cleaning.get("end").and_then(Value::as_str));
            if let (Some(cleaning_end), Some(after_time)) = (
                cleaning_end,
                after_time,
            ) {
                if after_time <= cleaning_end {
                    findings.push(
                        "/performance_comparison/after/observed_at",
                        "INCOMPARABLE",
                        "post-cleaning performance observation does not occur after cleaning completion",
                    );
                }
            }
        }
    }

    !findings.incomparable && !findings.invalid
}

fn check_record(record: &Value, findings: &mut Findings) -> (Option<f64>, usize, usize, Option<String>, bool) {
    if record.get("schema_version").and_then(Value::as_str) != Some("0.1.0") {
        findings.push("/schema_version", "INVALID", "unsupported biofouling evidence schema version");
    }
    if record.get("record_status").and_then(Value::as_str) == Some("synthetic") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "synthetic software fixture cannot establish actual hull condition, species status or cleaning performance",
        );
    }
    if record.get("record_status").and_then(Value::as_str) == Some("modelled") {
        findings.push(
            "/record_status",
            "INCOMPLETE",
            "modelled fouling state is not a measured hull inspection",
        );
    }

    let fingerprint = record.get("service_fingerprint").and_then(Value::as_str);
    if fingerprint.is_none_or(|value| value.len() != 64 || !value.chars().all(|c| c.is_ascii_hexdigit())) {
        findings.push("/service_fingerprint", "INVALID", "service fingerprint must be a 64-character hexadecimal digest");
    }

    let (coverage, zones, species) = check_inspection(record, findings);
    let cleaning = check_cleaning_event(record, findings);
    let performance_eligible = check_performance_comparison(record, cleaning.as_deref(), findings);

    if record.pointer("/coating_system/condition_status").and_then(Value::as_str) == Some("unknown") {
        findings.push(
            "/coating_system/condition_status",
            "INCOMPLETE",
            "coating condition is unknown",
        );
    }
    if record.pointer("/coating_system/system_family").and_then(Value::as_str) == Some("unknown") {
        findings.push(
            "/coating_system/system_family",
            "INCOMPLETE",
            "coating system family is unknown",
        );
    }

    (coverage, zones, species, cleaning, performance_eligible)
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: marine-biofouling-evidence-audit <schema.json> <record.json>".to_owned());
    }
    let schema = read_json(&args[1])?;
    let record = read_json(&args[2])?;
    let mut findings = Findings::default();
    validate_schema(&schema, &record, &mut findings);
    let (coverage, zones, species, cleaning_method, performance_eligible) =
        check_record(&record, &mut findings);

    let report = AuditReport {
        auditor: "marine-biofouling-evidence-audit",
        auditor_version: env!("CARGO_PKG_VERSION"),
        record_id: record.get("record_id").and_then(Value::as_str).map(str::to_owned),
        disposition: findings.disposition(),
        claim_ceiling: "Inspection/cleaning/performance evidence-contract checks only. This report does not identify organisms, authorize cleaning, prove zero environmental release, establish causal fuel savings, certify coatings or determine regulatory compliance.",
        hull_coverage_fraction: coverage,
        zone_count: zones,
        confirmed_species_count: species,
        cleaning_method,
        performance_comparison_eligible: performance_eligible,
        diagnostics: findings.diagnostics,
    };

    println!("{}", serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?);
    Ok(if report.disposition == "BIOFOULING_EVIDENCE_CONTRACT_VALID" { 0 } else { 2 })
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

    const BIOFOULING_SCHEMA: &str =
        include_str!("../../../../docs/research/marine/marine-biofouling-evidence-v0.1.0.schema.json");
    const BIOFOULING_FIXTURE: &str =
        include_str!("../../../../docs/research/marine/fixtures/synthetic-marine-biofouling-evidence-v0.1.0.json");

    fn fixture() -> Value {
        serde_json::from_str(BIOFOULING_FIXTURE).expect("biofouling fixture JSON")
    }

    #[test]
    fn committed_biofouling_schema_and_fixture_are_structurally_valid() {
        let schema: Value = serde_json::from_str(BIOFOULING_SCHEMA).expect("biofouling schema JSON");
        let record = fixture();
        assert!(jsonschema::draft202012::meta::is_valid(&schema));
        let validator = jsonschema::draft202012::options()
            .should_validate_formats(true)
            .build(&schema)
            .expect("biofouling schema compiles");
        assert!(validator.is_valid(&record));
    }

    #[test]
    fn synthetic_biofouling_fixture_is_never_physical_qualification() {
        let mut findings = Findings::default();
        check_record(&fixture(), &mut findings);
        assert!(findings.incomplete);
        assert!(!findings.invalid);
    }

    #[test]
    fn hull_coverage_must_match_inspected_area() {
        let mut record = fixture();
        record["inspection"]["declared_total_coverage_fraction"] = serde_json::json!(0.99);
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn duplicate_hull_zone_identity_is_invalid() {
        let mut record = fixture();
        let first = record["inspection"]["zones"][0].clone();
        record["inspection"]["zones"][1] = first;
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn unapproved_in_water_cleaning_is_blocked() {
        let mut record = fixture();
        record["cleaning_event"]["local_authorization_status"] = serde_json::json!("denied");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.blocked);
    }

    #[test]
    fn uncontrolled_in_water_cleaning_release_is_blocked() {
        let mut record = fixture();
        record["cleaning_event"]["containment_method"] = serde_json::json!("none");
        record["cleaning_event"]["waste_disposition"] = serde_json::json!("uncontrolled_release");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.blocked);
    }

    #[test]
    fn species_confirmation_requires_taxonomic_source() {
        let mut record = fixture();
        record["inspection"]["zones"][0]["species_status"] = serde_json::json!("confirmed");
        record["inspection"]["zones"][0]["taxonomic_identification"] =
            serde_json::json!("Synthetic taxon name");
        record["inspection"]["zones"][0]["species_evidence_source_id"] =
            serde_json::json!("MISSING-SOURCE");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.invalid);
    }

    #[test]
    fn matched_performance_pair_requires_identical_condition_fingerprint() {
        let mut record = fixture();
        record["performance_comparison"]["after"]["condition_fingerprint"] =
            serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.incomparable);
    }

    #[test]
    fn post_cleaning_performance_must_follow_cleaning_completion() {
        let mut record = fixture();
        record["performance_comparison"]["after"]["observed_at"] =
            serde_json::json!("2026-10-16T11:00:00Z");
        let mut findings = Findings::default();
        check_record(&record, &mut findings);
        assert!(findings.incomparable);
    }
}
