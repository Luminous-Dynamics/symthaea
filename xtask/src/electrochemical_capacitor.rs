//! Fail-closed structural and evidence audit for the electrochemical-capacitor v0.1 fixture.
//!
//! This is an evidence-contract auditor, not an electrochemical model, JSON-Schema implementation,
//! laboratory controller, safety assessor, or device-performance evaluator.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

const SCHEMA_PATH: &str =
    "schemas/energy/electrochemical_capacitor_benchmark_v0_1.schema.json";
const FIXTURE_PATH: &str = "fixtures/energy/electrochemical-capacitor/benchmark_v0_1.json";
const SOURCE_PATH: &str = "research/electrochemical-capacitor/source-register-v0.1.json";

const DEVICE_CLASSES: &[&str] = &[
    "edlc",
    "pseudocapacitor",
    "asymmetric_hybrid_capacitor",
    "zinc_ion_hybrid_supercapacitor",
    "lithium_ion_capacitor",
    "dielectric_capacitor",
];
const BOUNDARIES: &[&str] = &[
    "active_material",
    "electrode",
    "complete_device",
    "module",
    "installed_system",
];
const EVIDENCE_CLASSES: &[&str] = &[
    "synthetic",
    "assumption",
    "vendor",
    "literature",
    "surrogate",
    "calculation",
    "physical_measurement",
    "independent_laboratory",
];
const ADMISSIONS: &[&str] = &["candidate", "evaluation_admitted", "blocked", "unknown"];
const DISPOSITIONS: &[&str] = &[
    "candidate",
    "blocked_missing_complete_device_boundary",
    "measurement_only",
    "evaluation_eligible",
    "invalid",
    "failed",
    "censored",
];
const METRIC_STATUSES: &[&str] = &[
    "measured",
    "vendor_reported",
    "literature_reported",
    "synthetic",
    "unknown",
    "invalid",
    "censored",
];

fn required<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    value
        .get(key)
        .with_context(|| format!("{context}: missing required field `{key}`"))
}

fn required_str<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    let text = required(value, key, context)?
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))?;
    if text.trim().is_empty() {
        bail!("{context}.{key} must not be empty");
    }
    Ok(text)
}

fn one_of(value: &Value, key: &str, context: &str, allowed: &[&str]) -> Result<String> {
    let actual = required_str(value, key, context)?;
    if !allowed.contains(&actual) {
        bail!("{context}.{key} has unsupported value `{actual}`");
    }
    Ok(actual.to_string())
}

fn boundary_rank(boundary: &str) -> Option<u8> {
    match boundary {
        "active_material" => Some(0),
        "electrode" => Some(1),
        "complete_device" => Some(2),
        "module" => Some(3),
        "installed_system" => Some(4),
        _ => None,
    }
}

fn required_components_for_boundary(boundary: &str) -> Vec<&'static str> {
    let mut components = vec![
        "positive_electrode",
        "negative_electrode",
        "electrolyte",
        "separator",
        "current_collectors",
        "binder_additives",
        "tabs",
        "seals",
        "case",
    ];
    if boundary == "module" || boundary == "installed_system" {
        components.extend(["busbars", "module_enclosure", "monitoring_balancing"]);
    }
    if boundary == "installed_system" {
        components.extend([
            "power_conversion",
            "thermal_management",
            "protection",
            "wiring",
            "system_enclosure",
            "controls",
        ]);
    }
    components
}

fn normalized_metric_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace(' ', "_")
        .replace('-', "_")
}

fn has_measured_density_metric(
    metrics: &[Value],
    top_level_boundary: &str,
    names: &[&str],
    units: &[&str],
) -> bool {
    let Some(minimum_boundary) = boundary_rank(top_level_boundary) else {
        return false;
    };
    metrics.iter().any(|metric| {
        let name = metric["name"]
            .as_str()
            .map(normalized_metric_name)
            .unwrap_or_default();
        let unit = metric["unit"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .replace(' ', "");
        let metric_boundary = metric["boundary"]
            .as_str()
            .and_then(boundary_rank);
        metric["status"].as_str() == Some("measured")
            && metric["value"].as_f64().is_some_and(|value| value.is_finite() && value >= 0.0)
            && metric_boundary.is_some_and(|boundary| boundary == minimum_boundary)
            && names.contains(&name.as_str())
            && units.contains(&unit.as_str())
    })
}

fn valid_sha256(value: &Value) -> bool {
    let Some(digest) = value.as_str() else {
        return false;
    };
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn optional_sha256(value: &Value, context: &str) -> Result<()> {
    if !value.is_null() && !valid_sha256(value) {
        bail!("{context} must be null or `sha256:` followed by 64 lowercase hex characters");
    }
    Ok(())
}

fn audit_schema(root: &Path) -> Result<()> {
    let path = root.join(SCHEMA_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read schema at {}", path.display()))?;
    let schema: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("schema at {} is not valid JSON", path.display()))?;
    if schema["$schema"].as_str() != Some("https://json-schema.org/draft/2020-12/schema") {
        bail!("capacitor schema must declare JSON Schema draft 2020-12");
    }
    if schema["properties"]["records"]["items"]["$ref"].as_str()
        != Some("#/$defs/record")
    {
        bail!("capacitor schema records must reference the canonical record definition");
    }
    let required_fields = schema["$defs"]["record"]["required"]
        .as_array()
        .context("capacitor schema has no required-field contract for records")?;
    if !required_fields.iter().any(|field| field.as_str() == Some("denominator"))
        || schema["$defs"]["record"]["properties"]["denominator"].is_null()
    {
        bail!("capacitor schema must require the explicit mass/volume denominator contract");
    }
    Ok(())
}

fn audit_record(record: &Value, seen_ids: &mut BTreeSet<String>,
                seen_serials: &mut BTreeSet<String>) -> Result<()> {
    let context = required_str(record, "record_id", "record")?.to_string();
    if !seen_ids.insert(context.clone()) {
        bail!("duplicate record_id `{context}`");
    }
    if required_str(record, "schema_version", &context)? != "0.1.0" {
        bail!("{context}: unsupported record schema_version");
    }
    let device_class = one_of(record, "device_class", &context, DEVICE_CLASSES)?;
    let boundary = one_of(record, "boundary", &context, BOUNDARIES)?;
    let disposition = one_of(record, "disposition", &context, DISPOSITIONS)?;

    let denominator = required(record, "denominator", &context)?;
    let mass_kg = required(denominator, "mass_kg", &context)?;
    if !mass_kg.is_null()
        && mass_kg
            .as_f64()
            .is_none_or(|value| !value.is_finite() || value <= 0.0)
    {
        bail!("{context}.denominator.mass_kg must be null or finite and positive");
    }
    let volume_l = required(denominator, "volume_l", &context)?;
    if !volume_l.is_null()
        && volume_l
            .as_f64()
            .is_none_or(|value| !value.is_finite() || value <= 0.0)
    {
        bail!("{context}.denominator.volume_l must be null or finite and positive");
    }
    let denominator_basis = one_of(
        denominator,
        "basis",
        &context,
        &[
            "active_material_only",
            "electrode_basis",
            "complete_device_weighed",
            "module_vendor_reported",
            "module_measured",
            "installed_system_measured",
            "unknown",
        ],
    )?;
    let denominator_status = one_of(
        denominator,
        "status",
        &context,
        &["complete", "partial", "unknown"],
    )?;
    let included_components = required(denominator, "included_components", &context)?
        .as_array()
        .context("denominator.included_components must be an array")?;
    let excluded_components = required(denominator, "excluded_components", &context)?
        .as_array()
        .context("denominator.excluded_components must be an array")?;
    let mut included = BTreeSet::new();
    for component in included_components {
        let component = component
            .as_str()
            .context("included denominator component names must be strings")?;
        if component.trim().is_empty() || !included.insert(component) {
            bail!("{context}: included denominator components must be non-empty and unique");
        }
    }
    let mut excluded = BTreeSet::new();
    for component in excluded_components {
        let component = component
            .as_str()
            .context("excluded denominator component names must be strings")?;
        if component.trim().is_empty() || !excluded.insert(component) {
            bail!("{context}: excluded denominator components must be non-empty and unique");
        }
        if included.contains(component) {
            bail!("{context}: a denominator component cannot be both included and excluded");
        }
    }
    if denominator_status == "complete" {
        if mass_kg.is_null() || volume_l.is_null() {
            bail!("{context}: complete denominator requires measured mass_kg and volume_l");
        }
        if matches!(boundary.as_str(), "complete_device" | "module" | "installed_system")
            && !required_components_for_boundary(&boundary)
                .iter()
                .all(|component| included.contains(component))
        {
            bail!("{context}: denominator is missing a component required by its declared physical boundary");
        }
    }

    let design = required(record, "design", &context)?;
    for key in [
        "design_id",
        "revision",
        "positive_electrode_id",
        "negative_electrode_id",
        "electrolyte_id",
        "separator_id",
        "current_collector_id",
        "package_id",
        "process_id",
        "physical_device_serial",
    ] {
        required_str(design, key, &context)?;
    }
    let lots = required(design, "material_lot_ids", &context)?
        .as_array()
        .with_context(|| format!("{context}.design.material_lot_ids must be an array"))?;
    if lots.is_empty() {
        bail!("{context}: at least one material lot identity is required");
    }
    let mut unique_lots = BTreeSet::new();
    for lot in lots {
        let lot = lot.as_str().context("material lot IDs must be strings")?;
        if lot.trim().is_empty() || !unique_lots.insert(lot) {
            bail!("{context}: material lot IDs must be non-empty and unique");
        }
    }

    let serial = required_str(design, "physical_device_serial", &context)?;
    let placeholder = serial.to_ascii_lowercase().contains("not-provided")
        || serial.to_ascii_lowercase().contains("synthetic");
    if !placeholder && !seen_serials.insert(serial.to_string()) {
        bail!("duplicate physical device serial `{serial}`");
    }

    let conditions = required(record, "test_conditions", &context)?;
    for key in [
        "protocol_id",
        "current_profile_id",
        "fixture_id",
        "instrument_id",
        "calibration_id",
    ] {
        required_str(conditions, key, &context)?;
    }
    let v_min = required(conditions, "voltage_min_v", &context)?
        .as_f64()
        .context("voltage_min_v must be numeric")?;
    let v_max = required(conditions, "voltage_max_v", &context)?
        .as_f64()
        .context("voltage_max_v must be numeric")?;
    if v_min < 0.0 || v_max <= v_min {
        bail!("{context}: voltage window must satisfy 0 <= min < max");
    }
    required(conditions, "temperature_c", &context)?
        .as_f64()
        .context("temperature_c must be numeric")?;
    optional_sha256(required(conditions, "raw_data_digest", &context)?, &format!("{context}.raw_data_digest"))?;

    let evidence = required(record, "evidence", &context)?;
    required_str(evidence, "evidence_id", &context)?;
    let evidence_class = one_of(evidence, "class", &context, EVIDENCE_CLASSES)?;
    let admission = one_of(evidence, "admission", &context, ADMISSIONS)?;
    let independent_lab = required(evidence, "independent_lab", &context)?
        .as_bool()
        .context("independent_lab must be boolean")?;
    let source_uri = required(evidence, "source_uri", &context)?;
    if !source_uri.is_null()
        && !source_uri
            .as_str()
            .is_some_and(|uri| uri.starts_with("https://") || uri.starts_with("http://"))
    {
        bail!("{context}: source_uri must be null or an HTTP(S) URI");
    }
    optional_sha256(required(evidence, "sha256", &context)?, &format!("{context}.evidence.sha256"))?;

    if independent_lab && evidence_class != "independent_laboratory" {
        bail!("{context}: independent_lab=true requires independent_laboratory evidence class");
    }
    if admission == "evaluation_admitted"
        && (evidence_class == "synthetic" || evidence_class == "assumption")
    {
        bail!("{context}: synthetic/assumption evidence cannot be evaluation-admitted");
    }

    let metrics = required(record, "metrics", &context)?
        .as_array()
        .context("metrics must be an array")?;
    if metrics.is_empty() {
        bail!("{context}: at least one metric record is required");
    }
    for (index, metric) in metrics.iter().enumerate() {
        let mctx = format!("{context}.metrics[{index}]");
        required_str(metric, "name", &mctx)?;
        required_str(metric, "unit", &mctx)?;
        required_str(metric, "measurement_method", &mctx)?;
        one_of(metric, "boundary", &mctx, BOUNDARIES)?;
        let status = one_of(metric, "status", &mctx, METRIC_STATUSES)?;
        let metric_value = required(metric, "value", &mctx)?;
        if !metric_value.is_null() && !metric_value.is_number() {
            bail!("{mctx}.value must be numeric or null");
        }
        let uncertainty = required(metric, "uncertainty", &mctx)?;
        if !uncertainty.is_null()
            && uncertainty
                .as_f64()
                .is_none_or(|value| !value.is_finite() || value < 0.0)
        {
            bail!("{mctx}.uncertainty must be null or a finite non-negative number");
        }
        match status.as_str() {
            "measured" if evidence_class != "physical_measurement"
                && evidence_class != "independent_laboratory" =>
            {
                bail!("{mctx}: measured status requires physical measurement evidence");
            }
            "vendor_reported" if evidence_class != "vendor" => {
                bail!("{mctx}: vendor_reported status requires vendor evidence");
            }
            "literature_reported" if evidence_class != "literature" => {
                bail!("{mctx}: literature_reported status requires literature evidence");
            }
            "synthetic" if evidence_class != "synthetic" => {
                bail!("{mctx}: synthetic status requires synthetic evidence");
            }
            _ => {}
        }
    }

    let manufacturing = required(record, "manufacturing", &context)?;
    required_str(manufacturing, "batch_id", &context)?;
    required_str(manufacturing, "process_revision", &context)?;
    one_of(
        manufacturing,
        "yield_status",
        &context,
        &["not_measured", "measured", "failed", "unknown"],
    )?;
    let accepted_cost = required(manufacturing, "accepted_device_cost", &context)?;
    if !accepted_cost.is_null()
        && accepted_cost
            .as_f64()
            .is_none_or(|value| !value.is_finite() || value < 0.0)
    {
        bail!("{context}.accepted_device_cost must be null or a finite non-negative number");
    }

    if disposition == "evaluation_eligible" {
        if denominator_status != "complete" {
            bail!("{context}: evaluation_eligible requires a complete mass-and-volume denominator");
        }
        if !required_components_for_boundary(&boundary)
            .iter()
            .all(|component| included.contains(component))
        {
            bail!("{context}: evaluation_eligible requires component-complete accounting for its exact boundary");
        }
        let expected_basis = match boundary.as_str() {
            "complete_device" => "complete_device_weighed",
            "module" => "module_measured",
            "installed_system" => "installed_system_measured",
            _ => "unknown",
        };
        if denominator_basis != expected_basis {
            bail!("{context}: denominator basis does not match the exact evaluation boundary");
        }
        if device_class == "dielectric_capacitor" {
            bail!("{context}: dielectric capacitors require a separate device-class benchmark and denominator contract");
        }
        if admission != "evaluation_admitted"
            || (evidence_class != "physical_measurement"
                && evidence_class != "independent_laboratory")
        {
            bail!("{context}: evaluation_eligible requires admitted physical or independent-lab evidence");
        }
        if !matches!(boundary.as_str(), "complete_device" | "module" | "installed_system") {
            bail!("{context}: evaluation_eligible requires complete-device, module or installed-system boundary");
        }
        if !valid_sha256(required(conditions, "raw_data_digest", &context)?) {
            bail!("{context}: evaluation_eligible requires a raw-data SHA-256 digest");
        }
        if !valid_sha256(required(evidence, "sha256", &context)?) {
            bail!("{context}: evaluation_eligible requires an evidence-package SHA-256 digest");
        }
        let energy_names = ["specific_energy_density", "energy_density", "usable_energy_density"];
        let energy_units = ["wh/kg", "wh/l"];
        if !has_measured_density_metric(metrics, &boundary, &energy_names, &energy_units) {
            bail!("{context}: evaluation_eligible requires a measured device-bound energy-density metric");
        }
        let power_names = ["specific_power_density", "power_density"];
        let power_units = ["w/kg", "w/l", "kw/kg", "kw/l"];
        if !has_measured_density_metric(metrics, &boundary, &power_names, &power_units) {
            bail!("{context}: evaluation_eligible requires a measured device-bound power-density metric");
        }
        if placeholder {
            bail!("{context}: evaluation_eligible requires a unique physical device serial");
        }
    }
    if evidence_class == "synthetic" && disposition == "evaluation_eligible" {
        bail!("{context}: synthetic evidence cannot be promoted to evaluation_eligible");
    }
    if boundary == "active_material" && disposition == "evaluation_eligible" {
        bail!("{context}: active-material boundary cannot qualify a complete device");
    }
    Ok(())
}

fn audit_fixture_value(fixture: &Value) -> Result<usize> {
    if required_str(fixture, "schema_version", "fixture")? != "0.1.0" {
        bail!("fixture schema_version must be 0.1.0");
    }
    let records = required(fixture, "records", "fixture")?
        .as_array()
        .context("fixture.records must be an array")?;
    if records.is_empty() {
        bail!("fixture.records must not be empty");
    }
    let mut seen_ids = BTreeSet::new();
    let mut seen_serials = BTreeSet::new();
    for record in records {
        audit_record(record, &mut seen_ids, &mut seen_serials)?;
    }
    Ok(records.len())
}

fn audit_sources(root: &Path) -> Result<usize> {
    let path = root.join(SOURCE_PATH);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("could not read source register at {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("source register at {} is not valid JSON", path.display()))?;
    if required_str(&value, "schema_version", "source-register")? != "0.1.0" {
        bail!("source register schema_version must be 0.1.0");
    }
    let sources = required(&value, "sources", "source-register")?
        .as_array()
        .context("source-register.sources must be an array")?;
    if sources.len() < 10 {
        bail!("source register unexpectedly contains fewer than ten records");
    }
    let mut ids = BTreeSet::new();
    for source in sources {
        let id = required_str(source, "id", "source-register.sources")?;
        required_str(source, "title", id)?;
        let url = required_str(source, "url", id)?;
        if !url.starts_with("https://") && !url.starts_with("http://") {
            bail!("{id}: source URL must use HTTP(S)");
        }
        required_str(source, "publisher", id)?;
        required_str(source, "evidence_class", id)?;
        required_str(source, "permitted_use", id)?;
        required_str(source, "prohibited_inference", id)?;
        if !ids.insert(id.to_string()) {
            bail!("duplicate source-register ID `{id}`");
        }
    }
    Ok(sources.len())
}

pub fn audit(root: &Path) -> Result<()> {
    audit_schema(root)?;
    let fixture_path = root.join(FIXTURE_PATH);
    let bytes = std::fs::read(&fixture_path)
        .with_context(|| format!("could not read fixture at {}", fixture_path.display()))?;
    let fixture: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("fixture at {} is not valid JSON", fixture_path.display()))?;
    let record_count = audit_fixture_value(&fixture)?;
    let source_count = audit_sources(root)?;
    println!(
        "electrochemical-capacitor-audit: PASS ({} records; {} source records; contract semantics only)",
        record_count, source_count
    );
    println!(
        "claim ceiling: no physical performance, lifetime, safety, yield or manufacturing qualification"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask is one level below workspace root")
            .to_path_buf()
    }

    fn fixture() -> Value {
        let bytes = std::fs::read(root().join(FIXTURE_PATH)).expect("fixture must be readable");
        serde_json::from_slice(&bytes).expect("fixture must be valid JSON")
    }

    #[test]
    fn checked_in_fixture_and_source_register_pass() {
        audit(&root()).expect("checked-in capacitor contract should pass");
    }

    #[test]
    fn active_material_only_synthetic_claim_stays_blocked() {
        let value = fixture();
        let records = value["records"].as_array().expect("records array");
        let negative = &records[1];
        assert_eq!(negative["boundary"], "active_material");
        assert_eq!(negative["evidence"]["class"], "synthetic");
        assert_eq!(
            negative["disposition"],
            "blocked_missing_complete_device_boundary"
        );
        assert!(audit_fixture_value(&value).is_ok());
    }

    #[test]
    fn synthetic_evidence_cannot_be_promoted() {
        let mut record = test_only_promotable_record();
        record["evidence"]["class"] = json!("synthetic");
        let mut seen_ids = BTreeSet::new();
        let mut seen_serials = BTreeSet::new();
        assert!(audit_record(&record, &mut seen_ids, &mut seen_serials).is_err());
    }

    #[test]
    fn duplicate_physical_serial_is_rejected() {
        let mut value = fixture();
        value["records"][0]["design"]["physical_device_serial"] = json!("MODULE-REFERENCE-001");
        let mut duplicate = value["records"][0].clone();
        duplicate["record_id"] = json!("duplicate-eaton-record");
        value["records"].as_array_mut().unwrap().push(duplicate);
        assert!(audit_fixture_value(&value).is_err());
    }

    #[test]
    fn invalid_voltage_window_is_rejected() {
        let mut value = fixture();
        value["records"][0]["test_conditions"]["voltage_max_v"] = json!(0.0);
        assert!(audit_fixture_value(&value).is_err());
    }

    // This is an in-memory contract test record, not a physical measurement or checked-in fixture.
    fn test_only_promotable_record() -> Value {
        let mut record = fixture()["records"][1].clone();
        record["boundary"] = json!("complete_device");
        record["disposition"] = json!("evaluation_eligible");
        record["design"]["physical_device_serial"] = json!("TEST-ONLY-DEVICE-0001");
        record["design"]["material_lot_ids"] = json!(["TEST-ONLY-LOT"]);
        record["test_conditions"]["raw_data_digest"] = json!(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        record["evidence"]["evidence_id"] = json!("TEST-ONLY-EVIDENCE");
        record["evidence"]["class"] = json!("physical_measurement");
        record["evidence"]["admission"] = json!("evaluation_admitted");
        record["evidence"]["sha256"] = json!(
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        );
        record["evidence"]["source_uri"] = json!("https://example.invalid/test-only");
        record["denominator"] = json!({
            "mass_kg": 0.01,
            "volume_l": 0.001,
            "basis": "complete_device_weighed",
            "included_components": [
                "positive_electrode",
                "negative_electrode",
                "electrolyte",
                "separator",
                "current_collectors",
                "binder_additives",
                "tabs",
                "seals",
                "case"
            ],
            "excluded_components": [],
            "status": "complete"
        });
        record["manufacturing"]["batch_id"] = json!("TEST-ONLY-BATCH");
        record["manufacturing"]["process_revision"] = json!("TEST-ONLY-PROCESS");
        record["metrics"] = json!([
            {
                "name": "specific_energy_density",
                "value": 5.0,
                "unit": "Wh/kg",
                "boundary": "complete_device",
                "measurement_method": "test-only contract fixture",
                "uncertainty": 0.1,
                "status": "measured"
            },
            {
                "name": "specific_power_density",
                "value": 1000.0,
                "unit": "W/kg",
                "boundary": "complete_device",
                "measurement_method": "test-only contract fixture",
                "uncertainty": 10.0,
                "status": "measured"
            }
        ]);
        record
    }

    #[test]
    fn evaluation_eligibility_accepts_only_complete_device_bound_measurements() {
        let record = test_only_promotable_record();
        let mut seen_ids = BTreeSet::new();
        let mut seen_serials = BTreeSet::new();
        audit_record(&record, &mut seen_ids, &mut seen_serials)
            .expect("test-only complete-device control should satisfy the contract");
    }

    #[test]
    fn evaluation_eligibility_rejects_missing_device_power_density() {
        let mut record = test_only_promotable_record();
        let _ = record["metrics"].as_array_mut().unwrap().pop();
        let mut seen_ids = BTreeSet::new();
        let mut seen_serials = BTreeSet::new();
        assert!(audit_record(&record, &mut seen_ids, &mut seen_serials).is_err());
    }

    #[test]
    fn evaluation_eligibility_rejects_metric_from_a_different_boundary() {
        let mut record = test_only_promotable_record();
        record["metrics"][0]["boundary"] = json!("module");
        let mut seen_ids = BTreeSet::new();
        let mut seen_serials = BTreeSet::new();
        assert!(audit_record(&record, &mut seen_ids, &mut seen_serials).is_err());
    }



    #[test]
    fn complete_denominator_rejects_missing_volume_or_component_groups() {
        let mut record = test_only_promotable_record();
        record["denominator"]["volume_l"] = Value::Null;
        let mut seen_ids = BTreeSet::new();
        let mut seen_serials = BTreeSet::new();
        assert!(audit_record(&record, &mut seen_ids, &mut seen_serials).is_err());
    }

    #[test]
    fn vendor_claim_cannot_masquerade_as_independent_lab() {
        let mut value = fixture();
        value["records"][0]["evidence"]["independent_lab"] = json!(true);
        assert!(audit_fixture_value(&value).is_err());
    }
}
