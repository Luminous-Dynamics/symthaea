//! Per-run score validity disclosure.
//!
//! This strict disclosure binds a score to its model, harness, benchmark cohort,
//! outcome denominators, resource budget, regeneration provenance and exposure
//! controls. It distinguishes unknown from not-applicable and not-controlled.
//!
//! A valid disclosure is not proof of truth, complete logging, reviewer identity,
//! oracle correctness, or absence of training-data contamination. A controlled
//! acquired-exposure claim requires an independently captured boundary-monitor
//! receipt and explicit network/filesystem/tool/environment controls; in-agent
//! transcripts alone are insufficient.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const DISCLOSURE_SCHEMA: &str = "symthaea.score_validity_disclosure.v1";
pub const DISCLOSURE_REPORT_SCHEMA: &str = "symthaea.score_validity_validation_report.v1";
pub const VALIDATION_POLICY_VERSION: &str = "score-validity-policy-v1";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureStatusV1 {
    Controlled,
    NotControlled,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StringDisclosureV1 {
    pub status: DisclosureStatusV1,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DigestDisclosureV1 {
    pub status: DisclosureStatusV1,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CountDisclosureV1 {
    pub status: DisclosureStatusV1,
    pub value: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunIdentityV1 {
    pub run_id: String,
    pub model_provider: StringDisclosureV1,
    pub model_name: StringDisclosureV1,
    pub model_version: StringDisclosureV1,
    pub model_access_mode: StringDisclosureV1,
    pub harness_revision: DigestDisclosureV1,
    pub prompt_digest: DigestDisclosureV1,
    pub tool_manifest_digest: DigestDisclosureV1,
    pub environment_digest: DigestDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkBindingV1 {
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub cohort_id: String,
    pub cohort_created_at_utc: StringDisclosureV1,
    pub task_set_digest: DigestDisclosureV1,
    pub benchmark_manifest_digest: DigestDisclosureV1,
    pub score_artifact_digest: DigestDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreStratumV1 {
    pub stratum_id: String,
    pub task_family: StringDisclosureV1,
    pub source_family: StringDisclosureV1,
    pub task_count: u64,
    pub observed_count: u64,
    pub success_count: u64,
    pub score_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreMetricV1 {
    pub metric_id: String,
    pub metric_definition_digest: DigestDisclosureV1,
    pub strata: Vec<ScoreStratumV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElicitationBudgetV1 {
    pub model_calls: CountDisclosureV1,
    pub input_tokens: CountDisclosureV1,
    pub output_tokens: CountDisclosureV1,
    pub wall_clock_millis: CountDisclosureV1,
    pub retries: CountDisclosureV1,
    pub tool_calls: CountDisclosureV1,
    pub network_queries: CountDisclosureV1,
    pub human_assistance_minutes: CountDisclosureV1,
    pub compute_cost_microusd: CountDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThreatControlV1 {
    pub threat_class: String,
    pub status: DisclosureStatusV1,
    pub method_digest: DigestDisclosureV1,
    pub evidence_receipt_digest: DigestDisclosureV1,
    pub notes: StringDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryLayerV1 {
    pub status: DisclosureStatusV1,
    pub monitor_receipt_digest: DigestDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcquiredExposureV1 {
    pub status: DisclosureStatusV1,
    pub independent_monitor_receipt_digest: DigestDisclosureV1,
    pub network: BoundaryLayerV1,
    pub filesystem: BoundaryLayerV1,
    pub tools: BoundaryLayerV1,
    pub environment: BoundaryLayerV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerationDisclosureV1 {
    pub status: DisclosureStatusV1,
    pub generator_name: StringDisclosureV1,
    pub generator_version: StringDisclosureV1,
    pub source_revision: DigestDisclosureV1,
    pub configuration_digest: DigestDisclosureV1,
    pub seed_commitment_digest: DigestDisclosureV1,
    pub cohort_manifest_digest: DigestDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunArtifactDisclosureV1 {
    pub transcript_digest: DigestDisclosureV1,
    pub tool_trace_digest: DigestDisclosureV1,
    pub output_sanitization: DisclosureStatusV1,
    pub hidden_oracle_separation: DisclosureStatusV1,
    pub external_receipt_anchor: DigestDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreValidityDisclosureV1 {
    pub schema: String,
    pub disclosure_id: String,
    pub disclosure_created_at_utc: String,
    pub run_status: DisclosureStatusV1,
    pub identity: RunIdentityV1,
    pub benchmark: BenchmarkBindingV1,
    pub metrics: Vec<ScoreMetricV1>,
    pub elicitation_budget: ElicitationBudgetV1,
    /// Dataset-level direct / derivative / temporal / distributional controls.
    /// Acquired exposure is required separately below.
    pub dataset_threat_controls: Vec<ThreatControlV1>,
    pub acquired_exposure: AcquiredExposureV1,
    pub regeneration: RegenerationDisclosureV1,
    pub run_artifacts: RunArtifactDisclosureV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreValidityReportV1 {
    pub schema: String,
    pub validation_policy_version: String,
    pub disclosure_sha256: String,
    pub structurally_valid: bool,
    pub publication_ready: bool,
    pub confirmatory_eligible: bool,
    pub missing_or_unknown_field_count: u64,
    pub not_controlled_control_count: u64,
    pub unknown_control_count: u64,
    pub not_applicable_control_count: u64,
    pub controlled_control_count: u64,
    pub task_count_total: u64,
    pub observed_count_total: u64,
    pub score_metric_count: u64,
    pub acquired_exposure_status: DisclosureStatusV1,
    pub acquired_exposure_independently_monitored: bool,
    pub dataset_threat_controls_complete: bool,
    pub budget_complete: bool,
    pub regeneration_complete: bool,
    pub claim_boundary: String,
    pub issues: Vec<String>,
}

fn add_checked(total: &mut u64, value: u64, label: &str) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| format!("{label} overflows u64"))?;
    Ok(())
}

fn count_status(
    status: DisclosureStatusV1,
    controlled: &mut u64,
    not_controlled: &mut u64,
    not_applicable: &mut u64,
) -> Result<(), String> {
    match status {
        DisclosureStatusV1::Controlled => add_checked(controlled, 1, "controlled count"),
        DisclosureStatusV1::NotControlled => {
            add_checked(not_controlled, 1, "not-controlled count")
        }
        DisclosureStatusV1::Unknown => Ok(()),
        DisclosureStatusV1::NotApplicable => {
            add_checked(not_applicable, 1, "not-applicable count")
        }
    }
}

fn validate_text_status(
    status: DisclosureStatusV1,
    value: Option<&str>,
    label: &str,
    unknown_count: &mut u64,
) -> Result<(), String> {
    match status {
        DisclosureStatusV1::Controlled | DisclosureStatusV1::NotControlled => {
            let text = value.ok_or_else(|| format!("{label} requires a value"))?;
            if text.trim().is_empty() || text.len() > 2_048 {
                return Err(format!("{label} value must be nonempty and <=2048 bytes"));
            }
        }
        DisclosureStatusV1::Unknown => {
            add_checked(unknown_count, 1, "unknown field count")?;
            if value.is_some_and(|text| text.trim().is_empty() || text.len() > 2_048) {
                return Err(format!("{label} optional value must be nonempty and <=2048 bytes"));
            }
        }
        DisclosureStatusV1::NotApplicable => {
            if value.is_some() {
                return Err(format!("{label} must omit value when not_applicable"));
            }
        }
    }
    Ok(())
}

fn validate_digest_status(
    value: &DigestDisclosureV1,
    label: &str,
    unknown_count: &mut u64,
) -> Result<(), String> {
    match value.status {
        DisclosureStatusV1::Controlled | DisclosureStatusV1::NotControlled => {
            let digest = value.sha256.as_deref()
                .ok_or_else(|| format!("{label} requires a digest"))?;
            if !crate::valid_digest(digest) {
                return Err(format!("{label} must be a lowercase SHA-256 digest"));
            }
        }
        DisclosureStatusV1::Unknown => {
            add_checked(unknown_count, 1, "unknown field count")?;
            if let Some(digest) = value.sha256.as_deref()
                && !crate::valid_digest(digest)
            {
                return Err(format!("{label} optional digest is malformed"));
            }
        }
        DisclosureStatusV1::NotApplicable => {
            if value.sha256.is_some() {
                return Err(format!("{label} must omit digest when not_applicable"));
            }
        }
    }
    Ok(())
}

fn validate_count_status(
    value: &CountDisclosureV1,
    label: &str,
    unknown_count: &mut u64,
) -> Result<(), String> {
    match value.status {
        DisclosureStatusV1::Controlled | DisclosureStatusV1::NotControlled => {
            if value.value.is_none() {
                return Err(format!("{label} requires an observed count"));
            }
        }
        DisclosureStatusV1::Unknown => {
            add_checked(unknown_count, 1, "unknown field count")?;
        }
        DisclosureStatusV1::NotApplicable => {
            if value.value.is_some() {
                return Err(format!("{label} must omit value when not_applicable"));
            }
        }
    }
    Ok(())
}

fn validate_boundary_layer(
    layer: &BoundaryLayerV1,
    label: &str,
    unknown_count: &mut u64,
) -> Result<bool, String> {
    validate_digest_status(
        &layer.monitor_receipt_digest,
        &format!("{label}.monitor_receipt_digest"),
        unknown_count,
    )?;
    match layer.status {
        DisclosureStatusV1::Controlled => {
            if layer.monitor_receipt_digest.status != DisclosureStatusV1::Controlled {
                return Err(format!(
                    "{label} cannot be controlled without its monitor receipt digest"
                ));
            }
            Ok(true)
        }
        DisclosureStatusV1::NotControlled => Ok(false),
        DisclosureStatusV1::Unknown => {
            add_checked(unknown_count, 1, "unknown field count")?;
            Ok(false)
        }
        DisclosureStatusV1::NotApplicable => {
            if layer.monitor_receipt_digest.status != DisclosureStatusV1::NotApplicable {
                return Err(format!(
                    "{label} not_applicable status conflicts with its monitor receipt status"
                ));
            }
            Ok(false)
        }
    }
}

/// Validate structure and claim semantics for one run-bound disclosure.
///
/// Publication readiness means a complete, internally consistent disclosure
/// was supplied. Confirmatory eligibility is stricter: the run, all applicable
/// controls, budget, regeneration and acquired-exposure monitor evidence must
/// be known and controlled. Neither result certifies scientific truth.
pub fn validate_score_validity_disclosure(
    bytes: &[u8],
) -> Result<ScoreValidityReportV1, String> {
    let disclosure: ScoreValidityDisclosureV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("score disclosure does not match strict v1 JSON: {error}"))?;
    if disclosure.schema != DISCLOSURE_SCHEMA {
        return Err(format!("disclosure schema must be {DISCLOSURE_SCHEMA}"));
    }
    if disclosure.disclosure_id.trim().is_empty()
        || disclosure.disclosure_id.len() > 128
        || disclosure.run_status == DisclosureStatusV1::NotApplicable
    {
        return Err("disclosure_id is required and run_status cannot be not_applicable".to_owned());
    }
    if disclosure.disclosure_created_at_utc.trim().is_empty()
        || disclosure.disclosure_created_at_utc.len() > 64
        || disclosure.identity.run_id.trim().is_empty()
        || disclosure.identity.run_id.len() > 128
    {
        return Err("disclosure timestamp or run_id is empty/oversized".to_owned());
    }

    let mut unknown_count = 0_u64;
    let mut not_controlled_count = 0_u64;
    let mut not_applicable_count = 0_u64;
    let mut controlled_count = 0_u64;
    let mut missing_count = 0_u64;
    let mut issues = Vec::new();

    let text_fields = [
        (&disclosure.identity.model_provider, "identity.model_provider"),
        (&disclosure.identity.model_name, "identity.model_name"),
        (&disclosure.identity.model_version, "identity.model_version"),
        (&disclosure.identity.model_access_mode, "identity.model_access_mode"),
        (&disclosure.benchmark.cohort_created_at_utc, "benchmark.cohort_created_at_utc"),
        (&disclosure.regeneration.generator_name, "regeneration.generator_name"),
        (&disclosure.regeneration.generator_version, "regeneration.generator_version"),
    ];
    for (field, label) in text_fields {
        validate_text_status(field.status, field.value.as_deref(), label, &mut unknown_count)?;
        count_status(field.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if field.status == DisclosureStatusV1::Unknown {
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
    }

    let digest_fields = [
        (&disclosure.identity.harness_revision, "identity.harness_revision"),
        (&disclosure.identity.prompt_digest, "identity.prompt_digest"),
        (&disclosure.identity.tool_manifest_digest, "identity.tool_manifest_digest"),
        (&disclosure.identity.environment_digest, "identity.environment_digest"),
        (&disclosure.benchmark.task_set_digest, "benchmark.task_set_digest"),
        (&disclosure.benchmark.benchmark_manifest_digest, "benchmark.benchmark_manifest_digest"),
        (&disclosure.benchmark.score_artifact_digest, "benchmark.score_artifact_digest"),
        (&disclosure.regeneration.source_revision, "regeneration.source_revision"),
        (&disclosure.regeneration.configuration_digest, "regeneration.configuration_digest"),
        (&disclosure.regeneration.seed_commitment_digest, "regeneration.seed_commitment_digest"),
        (&disclosure.regeneration.cohort_manifest_digest, "regeneration.cohort_manifest_digest"),
        (&disclosure.run_artifacts.transcript_digest, "run_artifacts.transcript_digest"),
        (&disclosure.run_artifacts.tool_trace_digest, "run_artifacts.tool_trace_digest"),
        (&disclosure.run_artifacts.external_receipt_anchor, "run_artifacts.external_receipt_anchor"),
    ];
    for (field, label) in digest_fields {
        validate_digest_status(field, label, &mut unknown_count)?;
        count_status(field.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if field.status == DisclosureStatusV1::Unknown {
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
    }

    if disclosure.benchmark.benchmark_id.trim().is_empty()
        || disclosure.benchmark.benchmark_version.trim().is_empty()
        || disclosure.benchmark.cohort_id.trim().is_empty()
    {
        return Err("benchmark_id, benchmark_version and cohort_id are required".to_owned());
    }
    if disclosure.metrics.is_empty() {
        return Err("at least one score metric is required".to_owned());
    }

    let mut metric_ids = BTreeSet::new();
    let mut task_count_total = 0_u64;
    let mut observed_count_total = 0_u64;
    for metric in &disclosure.metrics {
        if metric.metric_id.trim().is_empty()
            || metric.metric_id.len() > 128
            || !metric_ids.insert(metric.metric_id.as_str())
        {
            return Err("metric IDs must be unique, nonempty and <=128 bytes".to_owned());
        }
        validate_digest_status(
            &metric.metric_definition_digest,
            "metric.metric_definition_digest",
            &mut unknown_count,
        )?;
        count_status(
            metric.metric_definition_digest.status,
            &mut controlled_count,
            &mut not_controlled_count,
            &mut not_applicable_count,
        )?;
        if metric.metric_definition_digest.status == DisclosureStatusV1::Unknown {
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
        if metric.strata.is_empty() {
            return Err(format!("metric {} must include at least one stratum", metric.metric_id));
        }
        let mut stratum_ids = BTreeSet::new();
        for stratum in &metric.strata {
            if stratum.stratum_id.trim().is_empty()
                || stratum.stratum_id.len() > 128
                || !stratum_ids.insert(stratum.stratum_id.as_str())
            {
                return Err(format!("metric {} contains duplicate/invalid stratum ID", metric.metric_id));
            }
            if stratum.task_count == 0 || stratum.observed_count > stratum.task_count {
                return Err(format!("metric {} stratum {} has invalid denominator/observed count", metric.metric_id, stratum.stratum_id));
            }
            if stratum.success_count > stratum.observed_count || stratum.score_permille > 1_000 {
                return Err(format!("metric {} stratum {} has invalid success count or score", metric.metric_id, stratum.stratum_id));
            }
            validate_text_status(stratum.task_family.status, stratum.task_family.value.as_deref(), "stratum.task_family", &mut unknown_count)?;
            validate_text_status(stratum.source_family.status, stratum.source_family.value.as_deref(), "stratum.source_family", &mut unknown_count)?;
            count_status(stratum.task_family.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
            count_status(stratum.source_family.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
            if stratum.task_family.status == DisclosureStatusV1::Unknown || stratum.source_family.status == DisclosureStatusV1::Unknown {
                add_checked(&mut missing_count, 1, "missing field count")?;
            }
            add_checked(&mut task_count_total, stratum.task_count, "total task count")?;
            add_checked(&mut observed_count_total, stratum.observed_count, "total observed count")?;
        }
    }

    let budget = &disclosure.elicitation_budget;
    let budget_fields = [
        (&budget.model_calls, "budget.model_calls"),
        (&budget.input_tokens, "budget.input_tokens"),
        (&budget.output_tokens, "budget.output_tokens"),
        (&budget.wall_clock_millis, "budget.wall_clock_millis"),
        (&budget.retries, "budget.retries"),
        (&budget.tool_calls, "budget.tool_calls"),
        (&budget.network_queries, "budget.network_queries"),
        (&budget.human_assistance_minutes, "budget.human_assistance_minutes"),
        (&budget.compute_cost_microusd, "budget.compute_cost_microusd"),
    ];
    let mut budget_complete = true;
    for (field, label) in budget_fields {
        validate_count_status(field, label, &mut unknown_count)?;
        count_status(field.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if field.status == DisclosureStatusV1::Unknown {
            budget_complete = false;
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
    }
    if let (Some(input), Some(output)) = (budget.input_tokens.value, budget.output_tokens.value)
        && input.checked_add(output).is_none()
    {
        return Err("input_tokens + output_tokens overflows u64".to_owned());
    }

    let required_dataset_threats = ["direct", "derivative", "temporal", "distributional"];
    let mut observed_threats = BTreeSet::new();
    let mut dataset_threat_controls_complete = true;
    for control in &disclosure.dataset_threat_controls {
        if !required_dataset_threats.contains(&control.threat_class.as_str()) {
            return Err(format!("unsupported dataset threat class {}", control.threat_class));
        }
        if !observed_threats.insert(control.threat_class.as_str()) {
            return Err(format!("duplicate dataset threat control {}", control.threat_class));
        }
        validate_digest_status(&control.method_digest, "threat.method_digest", &mut unknown_count)?;
        validate_digest_status(&control.evidence_receipt_digest, "threat.evidence_receipt_digest", &mut unknown_count)?;
        count_status(control.method_digest.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        count_status(control.evidence_receipt_digest.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        match control.status {
            DisclosureStatusV1::Controlled => {
                if control.method_digest.status != DisclosureStatusV1::Controlled
                    || control.evidence_receipt_digest.status != DisclosureStatusV1::Controlled
                {
                    return Err(format!("controlled {} threat requires controlled method and evidence digests", control.threat_class));
                }
            }
            DisclosureStatusV1::Unknown => {
                dataset_threat_controls_complete = false;
                add_checked(&mut unknown_count, 1, "unknown field count")?;
                add_checked(&mut missing_count, 1, "missing field count")?;
            }
            DisclosureStatusV1::NotControlled => {
                dataset_threat_controls_complete = false;
            }
            DisclosureStatusV1::NotApplicable => {
                if control.method_digest.status != DisclosureStatusV1::NotApplicable
                    || control.evidence_receipt_digest.status != DisclosureStatusV1::NotApplicable
                {
                    return Err(format!("not_applicable {} threat requires not_applicable evidence fields", control.threat_class));
                }
            }
        }
        count_status(control.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        validate_text_status(control.notes.status, control.notes.value.as_deref(), "threat.notes", &mut unknown_count)?;
        count_status(control.notes.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
    }
    if observed_threats.len() != required_dataset_threats.len()
        || required_dataset_threats.iter().any(|value| !observed_threats.contains(value))
    {
        return Err("exactly one direct, derivative, temporal and distributional threat control is required".to_owned());
    }

    let acquired = &disclosure.acquired_exposure;
    let network_ok = validate_boundary_layer(&acquired.network, "acquired.network", &mut unknown_count)?;
    let filesystem_ok = validate_boundary_layer(&acquired.filesystem, "acquired.filesystem", &mut unknown_count)?;
    let tools_ok = validate_boundary_layer(&acquired.tools, "acquired.tools", &mut unknown_count)?;
    let environment_ok = validate_boundary_layer(&acquired.environment, "acquired.environment", &mut unknown_count)?;
    validate_digest_status(&acquired.independent_monitor_receipt_digest, "acquired.independent_monitor_receipt_digest", &mut unknown_count)?;
    count_status(acquired.independent_monitor_receipt_digest.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
    for layer_status in [acquired.network.status, acquired.filesystem.status, acquired.tools.status, acquired.environment.status] {
        count_status(layer_status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
    }
    let acquired_exposure_independently_monitored =
        acquired.independent_monitor_receipt_digest.status == DisclosureStatusV1::Controlled;
    let all_boundaries_controlled = network_ok && filesystem_ok && tools_ok && environment_ok;
    match acquired.status {
        DisclosureStatusV1::Controlled => {
            if !acquired_exposure_independently_monitored || !all_boundaries_controlled {
                return Err("acquired exposure cannot be controlled without an independent monitor receipt and controlled network/filesystem/tool/environment boundaries".to_owned());
            }
        }
        DisclosureStatusV1::NotControlled => {}
        DisclosureStatusV1::Unknown => {
            add_checked(&mut unknown_count, 1, "unknown field count")?;
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
        DisclosureStatusV1::NotApplicable => {
            if [acquired.network.status, acquired.filesystem.status, acquired.tools.status, acquired.environment.status]
                .iter().any(|status| *status != DisclosureStatusV1::NotApplicable)
                || acquired.independent_monitor_receipt_digest.status != DisclosureStatusV1::NotApplicable
            {
                return Err("acquired exposure cannot be not_applicable when any boundary/monitor is applicable".to_owned());
            }
        }
    }
    count_status(acquired.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
    if acquired.status == DisclosureStatusV1::Unknown {
        dataset_threat_controls_complete = false;
    }

    let regeneration = &disclosure.regeneration;
    let mut regeneration_complete = regeneration.status == DisclosureStatusV1::Controlled;
    for (field, label) in [
        (&regeneration.generator_name, "regeneration.generator_name"),
        (&regeneration.generator_version, "regeneration.generator_version"),
    ] {
        validate_text_status(field.status, field.value.as_deref(), label, &mut unknown_count)?;
        count_status(field.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if field.status == DisclosureStatusV1::Unknown {
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
        if field.status != DisclosureStatusV1::Controlled {
            regeneration_complete = false;
        }
    }
    for (field, label) in [
        (&regeneration.source_revision, "regeneration.source_revision"),
        (&regeneration.configuration_digest, "regeneration.configuration_digest"),
        (&regeneration.seed_commitment_digest, "regeneration.seed_commitment_digest"),
        (&regeneration.cohort_manifest_digest, "regeneration.cohort_manifest_digest"),
    ] {
        validate_digest_status(field, label, &mut unknown_count)?;
        count_status(field.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if field.status == DisclosureStatusV1::Unknown {
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
        if field.status != DisclosureStatusV1::Controlled {
            regeneration_complete = false;
        }
    }
    if regeneration.status == DisclosureStatusV1::Controlled && !regeneration_complete {
        return Err("regeneration cannot be controlled without all generator/source/config/seed/cohort evidence".to_owned());
    }
    if regeneration.status == DisclosureStatusV1::Unknown {
        add_checked(&mut unknown_count, 1, "unknown field count")?;
        add_checked(&mut missing_count, 1, "missing field count")?;
        regeneration_complete = false;
    }
    if regeneration.status == DisclosureStatusV1::NotControlled {
        regeneration_complete = false;
    }
    count_status(regeneration.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;

    let artifacts = &disclosure.run_artifacts;
    for (field, label) in [
        (&artifacts.transcript_digest, "run_artifacts.transcript_digest"),
        (&artifacts.tool_trace_digest, "run_artifacts.tool_trace_digest"),
        (&artifacts.external_receipt_anchor, "run_artifacts.external_receipt_anchor"),
    ] {
        validate_digest_status(field, label, &mut unknown_count)?;
        count_status(field.status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if field.status == DisclosureStatusV1::Unknown {
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
    }
    for status in [artifacts.output_sanitization, artifacts.hidden_oracle_separation] {
        count_status(status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
        if status == DisclosureStatusV1::Unknown {
            add_checked(&mut unknown_count, 1, "unknown field count")?;
            add_checked(&mut missing_count, 1, "missing field count")?;
        }
    }

    if disclosure.run_status == DisclosureStatusV1::Unknown {
        add_checked(&mut unknown_count, 1, "unknown field count")?;
        add_checked(&mut missing_count, 1, "missing field count")?;
    } else {
        count_status(disclosure.run_status, &mut controlled_count, &mut not_controlled_count, &mut not_applicable_count)?;
    }

    let confirmatory_eligible = disclosure.run_status == DisclosureStatusV1::Controlled
        && dataset_threat_controls_complete
        && acquired.status == DisclosureStatusV1::Controlled
        && acquired_exposure_independently_monitored
        && budget_complete
        && regeneration_complete
        && not_controlled_count == 0
        && unknown_count == 0;

    if disclosure.run_status == DisclosureStatusV1::NotControlled {
        issues.push("run_status is not_controlled".to_owned());
    }
    if !budget_complete {
        issues.push("one or more elicitation-budget values are unknown".to_owned());
    }
    if !dataset_threat_controls_complete {
        issues.push("one or more threat controls are unknown or not controlled".to_owned());
    }
    if !regeneration_complete {
        issues.push("regeneration provenance is incomplete".to_owned());
    }
    if acquired.status != DisclosureStatusV1::Controlled {
        issues.push("acquired exposure is not independently controlled".to_owned());
    }

    let disclosure_sha256 = hash_domain(b"score-validity-disclosure.v1", bytes);
    Ok(ScoreValidityReportV1 {
        schema: DISCLOSURE_REPORT_SCHEMA.to_owned(),
        validation_policy_version: VALIDATION_POLICY_VERSION.to_owned(),
        disclosure_sha256,
        structurally_valid: true,
        publication_ready: true,
        confirmatory_eligible,
        missing_or_unknown_field_count: missing_count,
        not_controlled_control_count: not_controlled_count,
        unknown_control_count: unknown_count,
        not_applicable_control_count: not_applicable_count,
        controlled_control_count: controlled_count,
        task_count_total,
        observed_count_total,
        score_metric_count: u64::try_from(disclosure.metrics.len()).map_err(|_| "metric count overflows u64")?,
        acquired_exposure_status: acquired.status,
        acquired_exposure_independently_monitored,
        dataset_threat_controls_complete,
        budget_complete,
        regeneration_complete,
        claim_boundary: "syntactic_disclosure_validation_only_not_scientific_truth_not_training_data_absence_not_authenticated_runner_or_monitor_identity".to_owned(),
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn digest(label: &str) -> String {
        hash_domain(b"score-disclosure-test", label.as_bytes())
    }
    fn controlled_digest(label: &str) -> DigestDisclosureV1 {
        DigestDisclosureV1 { status: DisclosureStatusV1::Controlled, sha256: Some(digest(label)) }
    }
    fn controlled_text(value: &str) -> StringDisclosureV1 {
        StringDisclosureV1 { status: DisclosureStatusV1::Controlled, value: Some(value.to_owned()) }
    }
    fn controlled_count(value: u64) -> CountDisclosureV1 {
        CountDisclosureV1 { status: DisclosureStatusV1::Controlled, value: Some(value) }
    }
    fn controlled_boundary(label: &str) -> BoundaryLayerV1 {
        BoundaryLayerV1 { status: DisclosureStatusV1::Controlled, monitor_receipt_digest: controlled_digest(label) }
    }

    fn valid_disclosure() -> ScoreValidityDisclosureV1 {
        ScoreValidityDisclosureV1 {
            schema: DISCLOSURE_SCHEMA.to_owned(),
            disclosure_id: "disclosure-0001".to_owned(),
            disclosure_created_at_utc: "2026-10-31T12:00:00Z".to_owned(),
            run_status: DisclosureStatusV1::Controlled,
            identity: RunIdentityV1 {
                run_id: "run-0001".to_owned(),
                model_provider: controlled_text("provider-x"),
                model_name: controlled_text("model-y"),
                model_version: controlled_text("2026-10"),
                model_access_mode: controlled_text("api"),
                harness_revision: controlled_digest("harness"),
                prompt_digest: controlled_digest("prompt"),
                tool_manifest_digest: controlled_digest("tools"),
                environment_digest: controlled_digest("environment"),
            },
            benchmark: BenchmarkBindingV1 {
                benchmark_id: "benchmark-x".to_owned(),
                benchmark_version: "v1".to_owned(),
                cohort_id: "cohort-2026-10".to_owned(),
                cohort_created_at_utc: controlled_text("2026-10-30T00:00:00Z"),
                task_set_digest: controlled_digest("tasks"),
                benchmark_manifest_digest: controlled_digest("manifest"),
                score_artifact_digest: controlled_digest("scores"),
            },
            metrics: vec![ScoreMetricV1 {
                metric_id: "accuracy".to_owned(),
                metric_definition_digest: controlled_digest("accuracy-definition"),
                strata: vec![
                    ScoreStratumV1 {
                        stratum_id: "family-a".to_owned(),
                        task_family: controlled_text("family-a"),
                        source_family: controlled_text("source-1"),
                        task_count: 100,
                        observed_count: 100,
                        success_count: 90,
                        score_permille: 900,
                    },
                    ScoreStratumV1 {
                        stratum_id: "family-b".to_owned(),
                        task_family: controlled_text("family-b"),
                        source_family: controlled_text("source-2"),
                        task_count: 80,
                        observed_count: 80,
                        success_count: 64,
                        score_permille: 800,
                    },
                ],
            }],
            elicitation_budget: ElicitationBudgetV1 {
                model_calls: controlled_count(180),
                input_tokens: controlled_count(200_000),
                output_tokens: controlled_count(80_000),
                wall_clock_millis: controlled_count(900_000),
                retries: controlled_count(2),
                tool_calls: controlled_count(45),
                network_queries: controlled_count(0),
                human_assistance_minutes: controlled_count(0),
                compute_cost_microusd: controlled_count(50_000),
            },
            dataset_threat_controls: ["direct", "derivative", "temporal", "distributional"]
                .iter()
                .map(|threat_class| ThreatControlV1 {
                    threat_class: (*threat_class).to_owned(),
                    status: DisclosureStatusV1::Controlled,
                    method_digest: controlled_digest(&format!("{threat_class}-method")),
                    evidence_receipt_digest: controlled_digest(&format!("{threat_class}-receipt")),
                    notes: controlled_text("audited on frozen cohort"),
                })
                .collect(),
            acquired_exposure: AcquiredExposureV1 {
                status: DisclosureStatusV1::Controlled,
                independent_monitor_receipt_digest: controlled_digest("independent-monitor"),
                network: controlled_boundary("network-monitor"),
                filesystem: controlled_boundary("filesystem-monitor"),
                tools: controlled_boundary("tool-monitor"),
                environment: controlled_boundary("environment-monitor"),
            },
            regeneration: RegenerationDisclosureV1 {
                status: DisclosureStatusV1::Controlled,
                generator_name: controlled_text("challenge-generator"),
                generator_version: controlled_text("1.0.0"),
                source_revision: controlled_digest("generator-sha"),
                configuration_digest: controlled_digest("config"),
                seed_commitment_digest: controlled_digest("seed"),
                cohort_manifest_digest: controlled_digest("cohort"),
            },
            run_artifacts: RunArtifactDisclosureV1 {
                transcript_digest: controlled_digest("transcript"),
                tool_trace_digest: controlled_digest("tool-trace"),
                output_sanitization: DisclosureStatusV1::Controlled,
                hidden_oracle_separation: DisclosureStatusV1::Controlled,
                external_receipt_anchor: controlled_digest("receipt-anchor"),
            },
        }
    }

    fn validate(value: &ScoreValidityDisclosureV1) -> Result<ScoreValidityReportV1, String> {
        validate_score_validity_disclosure(&serde_json::to_vec(value).unwrap())
    }

    #[test]
    fn complete_disclosure_is_publication_ready_and_confirmatory_eligible() {
        let report = validate(&valid_disclosure()).unwrap();
        assert!(report.structurally_valid && report.publication_ready && report.confirmatory_eligible);
        assert_eq!(report.task_count_total, 180);
        assert_eq!(report.observed_count_total, 180);
        assert!(report.acquired_exposure_independently_monitored);
    }

    #[test]
    fn unknown_is_not_controlled_and_never_becomes_confirmatory_eligible() {
        let mut disclosure = valid_disclosure();
        disclosure.identity.model_version = StringDisclosureV1 { status: DisclosureStatusV1::Unknown, value: None };
        let report = validate(&disclosure).unwrap();
        assert!(report.publication_ready);
        assert!(!report.confirmatory_eligible);
        assert!(report.unknown_control_count > 0);
        assert!(report.missing_or_unknown_field_count > 0);
    }

    #[test]
    fn not_applicable_is_distinct_from_unknown_and_not_controlled() {
        let mut disclosure = valid_disclosure();
        disclosure.benchmark.cohort_created_at_utc = StringDisclosureV1 { status: DisclosureStatusV1::NotApplicable, value: None };
        let report = validate(&disclosure).unwrap();
        assert!(report.not_applicable_control_count > 0);
        assert_eq!(report.unknown_control_count, 0);
        assert_eq!(report.not_controlled_control_count, 0);
    }

    #[test]
    fn acquired_exposure_cannot_be_controlled_without_independent_monitor_receipt() {
        let mut disclosure = valid_disclosure();
        disclosure.acquired_exposure.independent_monitor_receipt_digest =
            DigestDisclosureV1 { status: DisclosureStatusV1::Unknown, sha256: None };
        assert!(validate(&disclosure).unwrap_err().contains("acquired exposure cannot be controlled"));
    }

    #[test]
    fn agent_transcript_cannot_override_unmonitored_network_boundary() {
        let mut disclosure = valid_disclosure();
        disclosure.acquired_exposure.network = BoundaryLayerV1 {
            status: DisclosureStatusV1::Unknown,
            monitor_receipt_digest: DigestDisclosureV1 { status: DisclosureStatusV1::Unknown, sha256: None },
        };
        disclosure.acquired_exposure.status = DisclosureStatusV1::Unknown;
        disclosure.run_artifacts.transcript_digest = controlled_digest("agent-says-no-network");
        let report = validate(&disclosure).unwrap();
        assert!(!report.confirmatory_eligible);
        assert!(!report.acquired_exposure_independently_monitored);
        assert_eq!(report.acquired_exposure_status, DisclosureStatusV1::Unknown);
    }

    #[test]
    fn missing_stratum_denominator_fails_closed() {
        let mut disclosure = valid_disclosure();
        disclosure.metrics[0].strata[0].task_count = 0;
        assert!(validate(&disclosure).unwrap_err().contains("invalid denominator"));
    }

    #[test]
    fn success_count_exceeding_observed_denominator_fails_closed() {
        let mut disclosure = valid_disclosure();
        disclosure.metrics[0].strata[0].success_count = 101;
        assert!(validate(&disclosure).unwrap_err().contains("invalid success count"));
    }

    #[test]
    fn regeneration_cannot_be_controlled_without_exact_generator_evidence() {
        let mut disclosure = valid_disclosure();
        disclosure.regeneration.configuration_digest = DigestDisclosureV1 { status: DisclosureStatusV1::Unknown, sha256: None };
        assert!(validate(&disclosure).unwrap_err().contains("regeneration cannot be controlled"));
    }

    #[test]
    fn all_four_dataset_threat_classes_are_required_and_separate_from_acquired() {
        let mut disclosure = valid_disclosure();
        disclosure.dataset_threat_controls.pop();
        assert!(validate(&disclosure).unwrap_err().contains("exactly one direct, derivative, temporal and distributional"));
    }

    #[test]
    fn unknown_fields_and_digest_mutations_are_rejected_or_bound() {
        let disclosure = valid_disclosure();
        let bytes = serde_json::to_vec(&disclosure).unwrap();
        let report = validate_score_validity_disclosure(&bytes).unwrap();
        let mut value: Value = serde_json::from_slice(&bytes).unwrap();
        value["unexpected_field"] = json!(true);
        assert!(validate_score_validity_disclosure(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut changed = disclosure;
        changed.benchmark.score_artifact_digest.sha256 = Some(digest("different-score"));
        let changed_report = validate_score_validity_disclosure(&serde_json::to_vec(&changed).unwrap()).unwrap();
        assert_ne!(report.disclosure_sha256, changed_report.disclosure_sha256);
    }

    #[test]
    fn controlled_threat_requires_method_and_evidence_digests() {
        let mut disclosure = valid_disclosure();
        disclosure.dataset_threat_controls[0].evidence_receipt_digest = DigestDisclosureV1 {
            status: DisclosureStatusV1::Unknown,
            sha256: None,
        };
        assert!(validate(&disclosure).unwrap_err().contains("controlled direct threat requires"));
    }

    #[test]
    fn malformed_digests_are_rejected_even_when_threat_is_not_controlled() {
        let mut disclosure = valid_disclosure();
        disclosure.dataset_threat_controls[0].status = DisclosureStatusV1::NotControlled;
        disclosure.dataset_threat_controls[0].method_digest.sha256 = Some("not-a-sha256".to_owned());
        let error = validate(&disclosure).unwrap_err();
        assert!(error.contains("threat.method_digest must be a lowercase SHA-256 digest"));
    }

    #[test]
    fn incomplete_budgets_are_disclosed_as_unknown_not_zero() {
        let mut disclosure = valid_disclosure();
        disclosure.elicitation_budget.network_queries = CountDisclosureV1 { status: DisclosureStatusV1::Unknown, value: None };
        let report = validate(&disclosure).unwrap();
        assert!(!report.confirmatory_eligible);
        assert!(report.unknown_control_count > 0);
        assert!(!report.budget_complete);
    }
}
