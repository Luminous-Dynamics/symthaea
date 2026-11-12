//! Offline structural audit for source and competitive-reference ledgers.
//!
//! This verifies metadata integrity and internal references only. It does not
//! re-run external experiments or independently authenticate a source claim.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRegister {
    schema_version: String,
    status: String,
    updated_on: String,
    created_for: Vec<String>,
    policy: SourcePolicy,
    records: Vec<SourceRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourcePolicy {
    source_url_is_not_qualification: bool,
    manufacturer_or_programme_statement_is_not_independent_verification: bool,
    review_article_values_require_primary_source_retrieval_before_matched_comparison: bool,
    unknown_values_must_remain_unknown: bool,
    cell_metrics_must_not_be_promoted_to_module_array_or_plant_metrics: bool,
    earth_orbit_array_metrics_must_not_be_promoted_to_lunar_surface_system_metrics: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRecord {
    source_id: String,
    title: String,
    publisher: String,
    published_or_updated: Option<String>,
    retrieved_on: Option<String>,
    updated_on: Option<String>,
    source_url: String,
    evidence_class: String,
    technology_scope: Vec<String>,
    reported_facts: Vec<String>,
    comparison_use: Vec<String>,
    not_established: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompetitiveSnapshot {
    schema_version: String,
    status: String,
    as_of_date: String,
    parent_issue: String,
    evidence_policy: EvidencePolicy,
    references: Vec<CompetitiveReference>,
    metrics: Vec<CompetitiveMetric>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidencePolicy {
    rules: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompetitiveReference {
    reference_id: String,
    source: String,
    url: String,
    source_class: String,
    published_year: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompetitiveMetric {
    metric_id: String,
    metric: String,
    unit: String,
    value: serde_json::Value,
    boundary: String,
    evidence_class: String,
    reference_id: String,
    caveats: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ReferenceAuditReport {
    pub report_schema: &'static str,
    pub status: &'static str,
    pub source_register_sha256: String,
    pub competitive_snapshot_sha256: String,
    pub source_record_count: usize,
    pub competitive_reference_count: usize,
    pub competitive_metric_count: usize,
    pub findings: Vec<ReferenceFinding>,
    pub claim_ceiling: &'static str,
}

#[derive(Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReferenceFinding {
    pub code: String,
    pub subject: String,
    pub message: String,
}

pub fn run(
    source_register_path: &Path,
    competitive_snapshot_path: &Path,
    output_path: Option<&Path>,
) -> Result<()> {
    let source_bytes = fs::read(source_register_path).with_context(|| {
        format!(
            "reading planetary-energy source register {}",
            source_register_path.display()
        )
    })?;
    let competitive_bytes = fs::read(competitive_snapshot_path).with_context(|| {
        format!(
            "reading planetary-energy competitive snapshot {}",
            competitive_snapshot_path.display()
        )
    })?;

    let report = audit_ledgers(&source_bytes, &competitive_bytes)?;
    let rendered = serde_json::to_string_pretty(&report)?;

    if let Some(path) = output_path {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating report directory {}", parent.display()))?;
            }
        }
        fs::write(path, &rendered)
            .with_context(|| {
                format!(
                    "writing planetary-energy reference report {}",
                    path.display()
                )
            })?;
    }
    println!("{rendered}");

    if report.status != "PASS" {
        bail!(
            "planetary-reference audit BLOCKED: {} finding(s)",
            report.findings.len()
        );
    }
    Ok(())
}

pub fn audit_ledgers(
    source_bytes: &[u8],
    competitive_bytes: &[u8],
) -> Result<ReferenceAuditReport> {
    let source: SourceRegister =
        serde_json::from_slice(source_bytes).context("parsing planetary-energy source register")?;
    let competitive: CompetitiveSnapshot = serde_json::from_slice(competitive_bytes)
        .context("parsing planetary-energy competitive snapshot")?;

    let mut findings = Vec::new();
    let mut source_ids = BTreeSet::new();
    let mut reference_ids = BTreeSet::new();
    let mut metric_ids = BTreeSet::new();

    if source.schema_version != "planetary-energy-source-register-v0.1" {
        findings.push(reference_finding(
            "SOURCE_SCHEMA",
            "source_register",
            "unexpected source-register schema version",
        ));
    }
    if source.status != "discovery_register_not_qualification" {
        findings.push(reference_finding(
            "SOURCE_STATUS",
            "source_register",
            "source register must retain discovery_register_not_qualification status",
        ));
    }
    if source.updated_on.trim().is_empty() {
        findings.push(reference_finding(
            "SOURCE_DATE",
            "source_register",
            "updated_on is required",
        ));
    }
    if source.created_for.is_empty()
        || source
            .created_for
            .iter()
            .any(|url| !url.starts_with("https://github.com/Luminous-Dynamics/symthaea/issues/"))
    {
        findings.push(reference_finding(
            "SOURCE_PARENT_LINKS",
            "source_register",
            "created_for must contain Symthaea issue URLs",
        ));
    }

    let policy = &source.policy;
    if !policy.source_url_is_not_qualification
        || !policy.manufacturer_or_programme_statement_is_not_independent_verification
        || !policy.review_article_values_require_primary_source_retrieval_before_matched_comparison
        || !policy.unknown_values_must_remain_unknown
        || !policy.cell_metrics_must_not_be_promoted_to_module_array_or_plant_metrics
        || !policy.earth_orbit_array_metrics_must_not_be_promoted_to_lunar_surface_system_metrics
    {
        findings.push(reference_finding(
            "SOURCE_POLICY_WEAKENED",
            "source_register",
            "all fail-closed source/evidence policy flags must remain true",
        ));
    }

    if source.records.is_empty() {
        findings.push(reference_finding(
            "SOURCE_REGISTER_EMPTY",
            "source_register",
            "at least one source record is required",
        ));
    }

    for record in &source.records {
        let subject = if record.source_id.trim().is_empty() {
            "<empty-source-id>"
        } else {
            record.source_id.as_str()
        };

        if record.source_id.trim().is_empty() || !source_ids.insert(record.source_id.as_str()) {
            findings.push(reference_finding(
                "SOURCE_IDENTITY",
                subject,
                "source_id must be non-empty and unique",
            ));
        }
        if record.title.trim().is_empty()
            || record.publisher.trim().is_empty()
            || record.evidence_class.trim().is_empty()
        {
            findings.push(reference_finding(
                "SOURCE_METADATA_INCOMPLETE",
                subject,
                "title, publisher and evidence_class are required",
            ));
        }
        if !record.source_url.starts_with("https://") {
            findings.push(reference_finding(
                "SOURCE_URL_INVALID",
                subject,
                "source_url must be an HTTPS URL",
            ));
        }
        if record.technology_scope.is_empty()
            || record.reported_facts.is_empty()
            || record.comparison_use.is_empty()
            || record.not_established.is_empty()
        {
            findings.push(reference_finding(
                "SOURCE_CLAIM_BOUNDARY_INCOMPLETE",
                subject,
                concat!(
                    "technology_scope, reported_facts, comparison_use and ",
                    "not_established must all be explicit"
                ),
            ));
        }
        if record
            .published_or_updated
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
            || record
                .retrieved_on
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            || record
                .updated_on
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
        {
            findings.push(reference_finding(
                "SOURCE_DATE_INVALID",
                subject,
                "optional date fields may be null but not blank",
            ));
        }
    }

    if competitive.schema_version != "planetary-energy-competitive-reference-v0.1" {
        findings.push(reference_finding(
            "COMPETITIVE_SCHEMA",
            "competitive_snapshot",
            "unexpected competitive snapshot schema version",
        ));
    }
    if competitive.status != "source_reported_benchmarks_not_luminous_results" {
        findings.push(reference_finding(
            "COMPETITIVE_STATUS",
            "competitive_snapshot",
            "competitive snapshot must not be labelled as Luminous results",
        ));
    }
    if competitive.as_of_date.trim().is_empty()
        || !competitive
            .parent_issue
            .ends_with("/issues/11756")
    {
        findings.push(reference_finding(
            "COMPETITIVE_LINEAGE",
            "competitive_snapshot",
            "as_of_date and parent issue #11756 must be explicit",
        ));
    }
    if competitive.evidence_policy.rules.is_empty()
        || competitive
            .evidence_policy
            .rules
            .iter()
            .any(|rule| rule.trim().is_empty())
    {
        findings.push(reference_finding(
            "COMPETITIVE_EVIDENCE_POLICY",
            "competitive_snapshot",
            "non-empty evidence-policy rules are required",
        ));
    }

    if competitive.references.is_empty() || competitive.metrics.is_empty() {
        findings.push(reference_finding(
            "COMPETITIVE_LEDGER_EMPTY",
            "competitive_snapshot",
            "at least one reference and one metric are required",
        ));
    }

    for reference in &competitive.references {
        let subject = if reference.reference_id.trim().is_empty() {
            "<empty-reference-id>"
        } else {
            reference.reference_id.as_str()
        };
        if reference.reference_id.trim().is_empty()
            || !reference_ids.insert(reference.reference_id.as_str())
        {
            findings.push(reference_finding(
                "REFERENCE_IDENTITY",
                subject,
                "reference_id must be non-empty and unique",
            ));
        }
        if reference.source.trim().is_empty()
            || reference.source_class.trim().is_empty()
            || reference.published_year == 0
        {
            findings.push(reference_finding(
                "REFERENCE_METADATA_INCOMPLETE",
                subject,
                "source, source_class and published_year are required",
            ));
        }
        if !reference.url.starts_with("https://") {
            findings.push(reference_finding(
                "REFERENCE_URL_INVALID",
                subject,
                "reference URL must be HTTPS",
            ));
        }
    }

    for metric in &competitive.metrics {
        let subject = if metric.metric_id.trim().is_empty() {
            "<empty-metric-id>"
        } else {
            metric.metric_id.as_str()
        };
        if metric.metric_id.trim().is_empty()
            || !metric_ids.insert(metric.metric_id.as_str())
        {
            findings.push(reference_finding(
                "METRIC_IDENTITY",
                subject,
                "metric_id must be non-empty and unique",
            ));
        }
        if metric.metric.trim().is_empty()
            || metric.unit.trim().is_empty()
            || metric.boundary.trim().is_empty()
            || metric.evidence_class.trim().is_empty()
            || metric.value.is_null()
            || metric.caveats.is_empty()
            || metric.caveats.iter().any(|caveat| caveat.trim().is_empty())
        {
            findings.push(reference_finding(
                "METRIC_BOUNDARY_INCOMPLETE",
                subject,
                "metric, unit, value, boundary, evidence_class and non-empty caveats are mandatory",
            ));
        }
        if !reference_ids.contains(metric.reference_id.as_str()) {
            findings.push(reference_finding(
                "DANGLING_METRIC_REFERENCE",
                subject,
                format!("unknown reference_id {}", metric.reference_id),
            ));
        }
    }

    findings.sort();
    let status = if findings.is_empty() { "PASS" } else { "BLOCKED" };

    Ok(ReferenceAuditReport {
        report_schema: "planetary-energy-reference-audit-report-v0.1",
        status,
        source_register_sha256: sha256_hex(source_bytes),
        competitive_snapshot_sha256: sha256_hex(competitive_bytes),
        source_record_count: source.records.len(),
        competitive_reference_count: competitive.references.len(),
        competitive_metric_count: competitive.metrics.len(),
        findings,
        claim_ceiling: "source_and_competitive_ledger_metadata_integrity_only",
    })
}

fn reference_finding(code: &str, subject: &str, message: impl Into<String>) -> ReferenceFinding {
    ReferenceFinding {
        code: code.to_string(),
        subject: subject.to_string(),
        message: message.into(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha2::Sha256::digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_REGISTER: &str =
        include_str!("../../research/planetary-energy/source-register-v0.1.json");
    const COMPETITIVE_SNAPSHOT: &str =
        include_str!(
            "../../benchmarks/planetary-energy-service/v0.1/competitive-reference-snapshot.json"
        );

    fn source_value() -> serde_json::Value {
        serde_json::from_str(SOURCE_REGISTER).expect("source register fixture parses")
    }

    fn competitive_value() -> serde_json::Value {
        serde_json::from_str(COMPETITIVE_SNAPSHOT).expect("competitive fixture parses")
    }

    #[test]
    fn checked_in_reference_ledgers_pass_metadata_audit() {
        let report = audit_ledgers(
            SOURCE_REGISTER.as_bytes(),
            COMPETITIVE_SNAPSHOT.as_bytes(),
        )
        .expect("reference ledgers parse");
        assert_eq!(report.status, "PASS");
        assert_eq!(report.source_record_count, 14);
        assert!(report.competitive_reference_count >= 10);
        assert!(report.competitive_metric_count >= 10);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn duplicate_source_identity_is_blocked() {
        let mut source = source_value();
        let first = source["records"][0].clone();
        source["records"][1]["source_id"] = first["source_id"].clone();
        let source_bytes = serde_json::to_vec(&source).expect("source serializes");
        let report = audit_ledgers(
            &source_bytes,
            COMPETITIVE_SNAPSHOT.as_bytes(),
        )
        .expect("reference ledgers parse");
        assert_eq!(report.status, "BLOCKED");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "SOURCE_IDENTITY"));
    }

    #[test]
    fn dangling_competitive_reference_is_blocked() {
        let mut competitive = competitive_value();
        competitive["metrics"][0]["reference_id"] =
            serde_json::json!("missing-reference");
        let competitive_bytes =
            serde_json::to_vec(&competitive).expect("competitive snapshot serializes");
        let report = audit_ledgers(
            SOURCE_REGISTER.as_bytes(),
            &competitive_bytes,
        )
        .expect("reference ledgers parse");
        assert_eq!(report.status, "BLOCKED");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "DANGLING_METRIC_REFERENCE"));
    }

    #[test]
    fn empty_metric_boundary_is_blocked() {
        let mut competitive = competitive_value();
        competitive["metrics"][0]["boundary"] = serde_json::json!("");
        let competitive_bytes =
            serde_json::to_vec(&competitive).expect("competitive snapshot serializes");
        let report = audit_ledgers(
            SOURCE_REGISTER.as_bytes(),
            &competitive_bytes,
        )
        .expect("reference ledgers parse");
        assert_eq!(report.status, "BLOCKED");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "METRIC_BOUNDARY_INCOMPLETE"));
    }

    #[test]
    fn source_claim_policy_cannot_be_weakened() {
        let mut source = source_value();
        source["policy"]["unknown_values_must_remain_unknown"] = serde_json::json!(false);
        let source_bytes = serde_json::to_vec(&source).expect("source register serializes");
        let report = audit_ledgers(
            &source_bytes,
            COMPETITIVE_SNAPSHOT.as_bytes(),
        )
        .expect("reference ledgers parse");
        assert_eq!(report.status, "BLOCKED");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "SOURCE_POLICY_WEAKENED"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut source = source_value();
        source["records"][0]["invented_performance_claim"] = serde_json::json!(99);
        let bytes = serde_json::to_vec(&source).expect("source register serializes");
        assert!(audit_ledgers(&bytes, COMPETITIVE_SNAPSHOT.as_bytes()).is_err());
    }
}
