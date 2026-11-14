//! Deterministic semantic audit for HMP-001 pluralistic knowledge records.
//!
//! This command checks the high-risk provenance and permission invariants. It is deliberately
//! not a general JSON Schema 2020-12 implementation; the adjacent schema remains the interchange
//! contract and must also be validated by a standards-compliant schema validator.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const VALID_PERMISSION_STATES: &[&str] =
    &["granted", "pending", "denied", "withdrawn", "not_requested"];
const VALID_ACCESS_CLASSES: &[&str] =
    &["public_source", "community_controlled", "restricted", "reference_only"];
const VALID_USES: &[&str] = &[
    "read",
    "display",
    "cite",
    "index",
    "retrieve",
    "summarize",
    "translate",
    "adapt",
    "train",
    "redistribute",
    "export",
    "preserve",
];

fn required_string<'a>(value: &'a Value, key: &str, at: &str) -> Result<&'a str> {
    let field = value
        .get(key)
        .with_context(|| format!("{at}.{key} is required"))?;
    let text = field
        .as_str()
        .with_context(|| format!("{at}.{key} must be a string"))?;
    if text.trim().is_empty() {
        bail!("{at}.{key} must not be empty");
    }
    Ok(text)
}

fn check_use_list(value: &Value, key: &str, at: &str) -> Result<BTreeSet<String>> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("{at}.{key} must be an array"))?;
    let mut checked = BTreeSet::new();
    for item in values {
        let use_name = item
            .as_str()
            .with_context(|| format!("{at}.{key} entries must be strings"))?;
        if !VALID_USES.contains(&use_name) {
            bail!("{at}.{key} contains unknown operation: {use_name}");
        }
        if !checked.insert(use_name.to_owned()) {
            bail!("{at}.{key} contains duplicate operation: {use_name}");
        }
    }
    Ok(checked)
}

fn check_schema_header(schema: &Value) -> Result<()> {
    let dialect = required_string(schema, "$schema", "schema")?;
    if dialect != "https://json-schema.org/draft/2020-12/schema" {
        bail!("schema.$schema must identify JSON Schema draft 2020-12");
    }
    let title = required_string(schema, "title", "schema")?;
    if title != "Pluralistic Knowledge Record" {
        bail!("schema.title is not the HMP-001 record schema");
    }
    required_string(schema, "$id", "schema")?;
    Ok(())
}

/// Check the HMP-001 high-risk semantic invariants for one record.
pub fn audit_record(record: &Value) -> Result<()> {
    let record_id = required_string(record, "record_id", "record")?;
    if !record_id.starts_with("hmp-") {
        bail!("record.record_id must start with hmp-");
    }
    if required_string(record, "schema_version", "record")? != "0.1.0" {
        bail!("record.schema_version must be 0.1.0");
    }

    let claim = record.get("claim").context("record.claim is required")?;
    required_string(claim, "text", "record.claim")?;
    required_string(claim, "claim_type", "record.claim")?;
    required_string(claim, "scope", "record.claim")?;

    let attribution = record
        .get("attribution")
        .context("record.attribution is required")?;
    required_string(attribution, "voice_type", "record.attribution")?;
    required_string(attribution, "scope_type", "record.attribution")?;
    required_string(attribution, "attribution_note", "record.attribution")?;

    let context = record.get("context").context("record.context is required")?;
    for key in ["time_scope", "place_scope", "language", "setting"] {
        required_string(context, key, "record.context")?;
    }

    let epistemic = record
        .get("epistemic")
        .context("record.epistemic is required")?;
    required_string(epistemic, "status", "record.epistemic")?;
    required_string(epistemic, "uncertainty_note", "record.epistemic")?;

    let provenance = record
        .get("provenance")
        .context("record.provenance is required")?;
    let sources = provenance
        .get("sources")
        .and_then(Value::as_array)
        .context("record.provenance.sources must be a non-empty array")?;
    if sources.is_empty() {
        bail!("record.provenance.sources must not be empty");
    }
    for (index, source) in sources.iter().enumerate() {
        let at = format!("record.provenance.sources[{index}]");
        required_string(source, "source_id", &at)?;
        required_string(source, "source_kind", &at)?;
        required_string(source, "source_reference", &at)?;
        required_string(source, "citation_note", &at)?;
    }

    let rights = record.get("rights").context("record.rights is required")?;
    let access_class = required_string(rights, "access_class", "record.rights")?;
    if !VALID_ACCESS_CLASSES.contains(&access_class) {
        bail!("record.rights.access_class is unknown: {access_class}");
    }
    let permission_status = required_string(rights, "permission_status", "record.rights")?;
    if !VALID_PERMISSION_STATES.contains(&permission_status) {
        bail!("record.rights.permission_status is unknown: {permission_status}");
    }
    required_string(rights, "permission_basis", "record.rights")?;

    let permitted = check_use_list(rights, "permitted_uses", "record.rights")?;
    let prohibited = check_use_list(rights, "prohibited_uses", "record.rights")?;
    if let Some(overlap) = permitted.intersection(&prohibited).next() {
        bail!("operation cannot be both permitted and prohibited: {overlap}");
    }

    if permission_status != "granted" && !permitted.is_empty() {
        bail!(
            "permission status {permission_status} is not granted; permitted_uses must be empty"
        );
    }

    if matches!(access_class, "community_controlled" | "restricted") {
        required_string(rights, "authority_reference", "record.rights")?;
        if permission_status == "granted" {
            required_string(rights, "permission_reference", "record.rights")?;
        }
    }

    let review = record.get("review").context("record.review is required")?;
    required_string(review, "status", "record.review")?;
    required_string(review, "review_note", "record.review")?;

    Ok(())
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

pub fn run(root: &Path, file: &Path, schema_path: &Path) -> Result<()> {
    let file_path = resolve(root, file);
    let schema_file = resolve(root, schema_path);

    let schema: Value = serde_json::from_slice(
        &std::fs::read(&schema_file)
            .with_context(|| format!("reading schema {}", schema_file.display()))?,
    )
    .with_context(|| format!("parsing schema {}", schema_file.display()))?;
    check_schema_header(&schema)?;

    let record: Value = serde_json::from_slice(
        &std::fs::read(&file_path)
            .with_context(|| format!("reading record {}", file_path.display()))?,
    )
    .with_context(|| format!("parsing record {}", file_path.display()))?;
    audit_record(&record)?;

    println!(
        "HMP-001 semantic audit: PASS ({})",
        required_string(&record, "record_id", "record")?
    );
    println!("Schema dialect/header: draft 2020-12");
    println!("Permission and provenance invariants: PASS");
    println!("Note: this is not full JSON Schema validation or a runtime authorization decision.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../research-fixtures/pluralistic-knowledge/synthetic-v0.1.0.json"
        ))
        .expect("synthetic HMP fixture must be valid JSON")
    }

    #[test]
    fn synthetic_fixture_passes_semantic_audit() {
        audit_record(&fixture()).expect("synthetic fixture should pass");
    }

    #[test]
    fn denied_permission_with_allowed_uses_is_rejected() {
        let mut record = fixture();
        record["rights"]["permission_status"] = json!("denied");
        assert!(audit_record(&record).is_err());
    }

    #[test]
    fn community_controlled_record_requires_authority() {
        let mut record = fixture();
        record["rights"]["access_class"] = json!("community_controlled");
        assert!(audit_record(&record).is_err());
    }

    #[test]
    fn empty_provenance_is_rejected() {
        let mut record = fixture();
        record["provenance"]["sources"] = json!([]);
        assert!(audit_record(&record).is_err());
    }

    #[test]
    fn permitted_and_prohibited_operations_cannot_overlap() {
        let mut record = fixture();
        record["rights"]["prohibited_uses"] = json!(["train"]);
        assert!(audit_record(&record).is_err());
    }

    #[test]
    fn pending_permission_with_no_uses_is_allowed() {
        let mut record = fixture();
        record["rights"]["permission_status"] = json!("pending");
        record["rights"]["permitted_uses"] = json!([]);
        audit_record(&record).expect("pending record may exist but cannot authorize use");
    }

    #[test]
    fn schema_header_uses_draft_2020_12() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../schemas/pluralistic-knowledge-record-v0.1.0.schema.json"
        ))
        .expect("schema must be valid JSON");
        check_schema_header(&schema).expect("schema header should match");
    }
}
