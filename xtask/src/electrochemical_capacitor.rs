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
    if schema["$defs"]["record"]["required"].as_array().is_none() {
        bail!("capacitor schema has no required-field contract for records");
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
    one_of(record, "device_class", &context, DEVICE_CLASSES)?;
    let boundary = one_of(record, "boundary", &context, BOUNDARIES)?;
    let disposition = one_of(record, "disposition", &context, DISPOSITIONS)?;

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
        let mut value = fixture();
        let negative = &mut value["records"][1];
        negative["boundary"] = json!("complete_device");
        negative["evidence"]["admission"] = json!("evaluation_admitted");
        negative["disposition"] = json!("evaluation_eligible");
        negative["test_conditions"]["raw_data_digest"] =
            json!("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert!(audit_fixture_value(&value).is_err());
    }

    #[test]
    fn duplicate_physical_serial_is_rejected() {
        let mut value = fixture();
        let first = value["records"][0].clone();
        let mut duplicate = first;
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

    #[test]
    fn vendor_claim_cannot_masquerade_as_independent_lab() {
        let mut value = fixture();
        value["records"][0]["evidence"]["independent_lab"] = json!(true);
        assert!(audit_fixture_value(&value).is_err());
    }
}
