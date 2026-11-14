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

    if access_class == "reference_only"
        && permitted.iter().any(|operation| operation != "cite")
    {
        bail!("reference_only records may authorize citation only");
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


const REQUIRED_BENCH_TRACKS: &[&str] = &[
    "HMP-ATTR",
    "HMP-SOURCE",
    "HMP-PLURAL",
    "HMP-UNCERTAINTY",
    "HMP-RIGHTS",
    "HMP-PHILO",
    "HMP-PSYCH",
    "HMP-SOC",
    "HMP-CORRECT",
];

fn non_empty_string_array<'a>(value: &'a Value, key: &str, at: &str) -> Result<Vec<&'a str>> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("{at}.{key} must be an array"))?;
    if values.is_empty() {
        bail!("{at}.{key} must not be empty");
    }
    values
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let text = item
                .as_str()
                .with_context(|| format!("{at}.{key}[{index}] must be a string"))?;
            if text.trim().is_empty() {
                bail!("{at}.{key}[{index}] must not be empty");
            }
            Ok(text)
        })
        .collect()
}

/// Validate benchmark fixture integrity and its rights-related negative control.
/// This does not evaluate model answers or establish benchmark validity.
pub fn audit_benchmark(fixture: &Value) -> Result<usize> {
    let benchmark_id = required_string(fixture, "benchmark_id", "benchmark")?;
    if !benchmark_id.starts_with("HMP-BENCH-") {
        bail!("benchmark.benchmark_id must start with HMP-BENCH-");
    }
    if required_string(fixture, "version", "benchmark")? != "0.1.0" {
        bail!("benchmark.version must be 0.1.0");
    }
    if required_string(fixture, "status", "benchmark")? != "synthetic_fixture_only" {
        bail!("benchmark.status must be synthetic_fixture_only");
    }
    required_string(fixture, "purpose", "benchmark")?;
    required_string(fixture, "scoring_warning", "benchmark")?;

    let cases = fixture
        .get("cases")
        .and_then(Value::as_array)
        .context("benchmark.cases must be a non-empty array")?;
    if cases.is_empty() {
        bail!("benchmark.cases must not be empty");
    }

    let mut case_ids = BTreeSet::new();
    let mut present_tracks = BTreeSet::new();

    for (index, case) in cases.iter().enumerate() {
        let at = format!("benchmark.cases[{index}]");
        let case_id = required_string(case, "case_id", &at)?;
        if !case_ids.insert(case_id.to_owned()) {
            bail!("duplicate benchmark case_id: {case_id}");
        }

        let track = required_string(case, "track", &at)?;
        if !REQUIRED_BENCH_TRACKS.contains(&track) {
            bail!("{at}.track is not a registered HMP-Bench v0.1 track: {track}");
        }
        present_tracks.insert(track.to_owned());

        required_string(case, "scenario", &at)?;
        required_string(case, "prompt", &at)?;
        required_string(case, "evaluator", &at)?;
        non_empty_string_array(case, "expected_behaviors", &at)?;
        non_empty_string_array(case, "forbidden_behaviors", &at)?;
        non_empty_string_array(case, "hard_fail_conditions", &at)?;

        let evidence = case
            .get("evidence")
            .and_then(Value::as_array)
            .with_context(|| format!("{at}.evidence must be a non-empty array"))?;
        if evidence.is_empty() {
            bail!("{at}.evidence must not be empty");
        }
        let mut evidence_ids = BTreeSet::new();
        for (evidence_index, item) in evidence.iter().enumerate() {
            let evidence_at = format!("{at}.evidence[{evidence_index}]");
            let source_id = required_string(item, "id", &evidence_at)?;
            if !evidence_ids.insert(source_id.to_owned()) {
                bail!("{evidence_at}.id is duplicated within case {case_id}");
            }
            required_string(item, "voice_type", &evidence_at)?;
            required_string(item, "scope", &evidence_at)?;
            required_string(item, "text", &evidence_at)?;
        }

        if track == "HMP-RIGHTS" {
            let rights = case
                .get("permission")
                .context("HMP-RIGHTS case must include permission")?;
            let access_class = required_string(rights, "access_class", "permission")?;
            if !matches!(access_class, "restricted" | "community_controlled") {
                bail!("HMP-RIGHTS case must exercise restricted or community-controlled access");
            }
            let status = required_string(rights, "permission_status", "permission")?;
            if !VALID_PERMISSION_STATES.contains(&status) {
                bail!("HMP-RIGHTS permission_status is unknown: {status}");
            }

            let allowed = check_use_list(rights, "permitted_uses", "permission")?;
            let prohibited = check_use_list(rights, "prohibited_uses", "permission")?;
            if let Some(overlap) = allowed.intersection(&prohibited).next() {
                bail!("HMP-RIGHTS operation cannot be both permitted and prohibited: {overlap}");
            }
            if status != "granted" && !allowed.is_empty() {
                bail!("HMP-RIGHTS non-granted permission must have zero permitted_uses");
            }
            // A denial or pending grant still needs a traceable authority basis.
            required_string(rights, "authority_reference", "permission")?;
            if status == "granted" {
                required_string(rights, "permission_reference", "permission")?;
            }

            let required_denied =
                non_empty_string_array(rights, "required_denied_operations", "permission")?;
            for operation in required_denied {
                if allowed.contains(operation) {
                    bail!("HMP-RIGHTS operation {operation} is marked required-denied but permitted");
                }
                if !prohibited.contains(operation) {
                    bail!("HMP-RIGHTS operation {operation} must appear in prohibited_uses");
                }
            }
        }
    }

    for required_track in REQUIRED_BENCH_TRACKS {
        if !present_tracks.contains(*required_track) {
            bail!("benchmark fixture is missing required track {required_track}");
        }
    }

    Ok(cases.len())
}

pub fn run_benchmark(root: &Path, file: &Path) -> Result<()> {
    let file_path = resolve(root, file);
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(&file_path)
            .with_context(|| format!("reading benchmark {}", file_path.display()))?,
    )
    .with_context(|| format!("parsing benchmark {}", file_path.display()))?;

    let case_count = audit_benchmark(&fixture)?;
    println!(
        "HMP-Bench fixture audit: PASS ({}; {} cases)",
        required_string(&fixture, "benchmark_id", "benchmark")?,
        case_count
    );
    println!("Required track coverage: PASS");
    println!("Synthetic-only declaration: PASS");
    println!("Rights negative control: PASS");
    println!("Note: this audits fixture integrity only; it does not score model outputs or validate cultural competence.");
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
    fn reference_only_record_rejects_non_citation_operations() {
        let mut record = fixture();
        record["rights"]["access_class"] = json!("reference_only");
        record["rights"]["permitted_uses"] = json!(["read"]);
        assert!(audit_record(&record).is_err());
    }

    #[test]
    fn reference_only_record_can_allow_citation_only() {
        let mut record = fixture();
        record["rights"]["access_class"] = json!("reference_only");
        record["rights"]["permitted_uses"] = json!(["cite"]);
        record["rights"]["prohibited_uses"] = json!([
            "read", "display", "index", "retrieve", "summarize", "translate",
            "adapt", "train", "redistribute", "export", "preserve"
        ]);
        audit_record(&record).expect("reference-only citation grant should pass");
    }


    #[test]
    fn synthetic_hmp_bench_fixture_has_unique_cases_and_expected_controls() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../research-fixtures/hmp-bench/synthetic-adversarial-v0.1.0.json"
        ))
        .expect("synthetic HMP-Bench fixture must be valid JSON");
        assert_eq!(
            fixture["benchmark_id"].as_str(),
            Some("HMP-BENCH-SYNTHETIC-ADVERSARIAL-001")
        );
        assert_eq!(fixture["status"].as_str(), Some("synthetic_fixture_only"));
        let cases = fixture["cases"]
            .as_array()
            .expect("benchmark cases must be an array");
        assert_eq!(cases.len(), 14, "synthetic starter fixture case count changed");

        let mut ids = BTreeSet::new();
        let mut tracks = BTreeSet::new();
        for case in cases {
            let id = case["case_id"]
                .as_str()
                .expect("each case needs a case_id");
            let track = case["track"]
                .as_str()
                .expect("each case needs a track");
            assert!(ids.insert(id.to_owned()), "duplicate case id: {id}");
            tracks.insert(track.to_owned());
            assert!(
                case["expected_behaviors"].as_array().is_some_and(|v| !v.is_empty()),
                "case {id} needs expected behaviors"
            );
            assert!(
                case["forbidden_behaviors"].as_array().is_some_and(|v| !v.is_empty()),
                "case {id} needs forbidden behaviors"
            );
            assert!(
                case["hard_fail_conditions"].as_array().is_some_and(|v| !v.is_empty()),
                "case {id} needs hard-fail conditions"
            );
        }
        assert_eq!(tracks.len(), 9, "each registered v0.1 track must be represented");
    }

    fn bench_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../research-fixtures/hmp-bench/synthetic-adversarial-v0.1.0.json"
        ))
        .expect("synthetic HMP-Bench fixture must be valid JSON")
    }

    #[test]
    fn hmp_bench_fixture_passes_integrity_audit() {
        assert_eq!(audit_benchmark(&bench_fixture()).expect("fixture should pass"), 14);
    }

    #[test]
    fn rights_negative_control_rejects_any_allowed_operation() {
        let mut fixture = bench_fixture();
        fixture["cases"][4]["permission"]["permitted_uses"] = json!(["read"]);
        assert!(audit_benchmark(&fixture).is_err());
    }

    #[test]
    fn hmp_bench_fixture_rejects_duplicate_case_ids() {
        let mut fixture = bench_fixture();
        fixture["cases"][1]["case_id"] = fixture["cases"][0]["case_id"].clone();
        assert!(audit_benchmark(&fixture).is_err());
    }

    #[test]
    fn hmp_bench_fixture_requires_all_registered_tracks() {
        let mut fixture = bench_fixture();
        fixture["cases"]
            .as_array_mut()
            .expect("cases array")
            .retain(|case| case["track"].as_str() != Some("HMP-CORRECT"));
        assert!(audit_benchmark(&fixture).is_err());
    }

    #[test]
    fn citation_only_grant_must_not_authorize_training() {
        let fixture = bench_fixture();
        let rights_case = fixture["cases"]
            .as_array()
            .expect("cases array")
            .iter()
            .find(|case| case["case_id"] == "HMP-RIGHTS-SYN-002")
            .expect("citation-only rights case");
        assert_eq!(rights_case["permission"]["permitted_uses"], json!(["cite"]));
        assert!(
            rights_case["permission"]["prohibited_uses"]
                .as_array()
                .expect("prohibited_uses array")
                .iter()
                .any(|operation| operation == "train")
        );
        audit_benchmark(&fixture).expect("scoped citation grant should pass fixture audit");
    }

    #[test]
    fn granted_community_rights_require_permission_reference() {
        let mut fixture = bench_fixture();
        fixture["cases"][9]["permission"]["permission_reference"] = Value::Null;
        assert!(audit_benchmark(&fixture).is_err());
    }

    #[test]
    fn denied_restricted_knowledge_still_requires_authority_reference() {
        let mut fixture = bench_fixture();
        fixture["cases"][4]["permission"]["authority_reference"] = Value::Null;
        assert!(audit_benchmark(&fixture).is_err());
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
