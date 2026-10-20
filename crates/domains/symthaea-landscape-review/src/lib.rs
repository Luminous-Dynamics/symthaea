// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Read-only, evidence-bound landscape review contract.
//!
//! This crate verifies cross-repository artifact references and summarizes
//! evidence coverage. It does not run landscape physics, rank alternatives,
//! infer causal effects, validate legal consent, or mutate project documents.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const REQUEST_SCHEMA: &str = "luminous.landscape-review-request.v0";
pub const REQUEST_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-review-request-v0.schema.json";
pub const REPORT_SCHEMA: &str = "luminous.landscape-review-report.v0";
pub const REPORT_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-review-report-v0.schema.json";
pub const PROJECT_SCHEMA: &str = "luminous.landscape-design.v0";
pub const PROJECT_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-design-v0.schema.json";
pub const EVALUATION_PLAN_SCHEMA: &str = "luminous.landscape-evaluation-plan.v0";
pub const EVALUATION_PLAN_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-evaluation-plan-v0.schema.json";
pub const SCENARIO_SCHEMA: &str = "luminous.landscape-scenario-manifest.v0";
pub const SCENARIO_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-scenario-manifest-v0.schema.json";
pub const OBSERVATION_BATCH_SCHEMA: &str = "luminous.landscape-observation-batch.v0";
pub const OBSERVATION_BATCH_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-observation-batch-v0.schema.json";
pub const STEWARDSHIP_SCHEMA: &str = "luminous.landscape-stewardship-profile.v0";
pub const STEWARDSHIP_SCHEMA_URI: &str =
    "https://luminousdynamics.io/schemas/landscape-stewardship-profile-v0.schema.json";
pub const LOCAL_FRAME: &str = "LOCAL_CARTESIAN_DRAWING_UNITS";
pub const MAX_ALTERNATIVES: usize = 8;
pub const MAX_REQUIRED_METRICS: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum LandscapeScope {
    SiteParcel,
    District,
    Watershed,
    Bioregion,
}

impl LandscapeScope {
    fn as_str(&self) -> &'static str {
        match self {
            Self::SiteParcel => "site-parcel",
            Self::District => "district",
            Self::Watershed => "watershed",
            Self::Bioregion => "bioregion",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactRole {
    ProjectSnapshot,
    EvaluationPlan,
    ScenarioManifest,
    ScenarioResult,
    ObservationBatch,
    StewardshipRecord,
    SourceDataset,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceClass {
    Observed,
    Curated,
    Synthetic,
    Modelled,
    Assumed,
    Missing,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MetricState {
    Available,
    Missing,
    NotComputable,
    Conflicting,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub role: ArtifactRole,
    pub sha256: String,
    pub schema_uri: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MetricEvidence {
    pub metric_id: String,
    pub unit: String,
    pub state: MetricState,
    pub evidence_class: EvidenceClass,
    pub method_uri: Option<String>,
    pub uncertainty_summary: String,
    pub supporting_artifact_sha256s: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AlternativeInput {
    pub alternative_id: String,
    pub project_snapshot: ArtifactRef,
    pub evaluation_plan: Option<ArtifactRef>,
    pub scenario_manifests: Vec<ArtifactRef>,
    pub scenario_results: Vec<ArtifactRef>,
    pub observation_batches: Vec<ArtifactRef>,
    pub stewardship_records: Vec<ArtifactRef>,
    pub source_datasets: Vec<ArtifactRef>,
    pub metrics: Vec<MetricEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewRequest {
    pub schema: String,
    pub schema_uri: String,
    pub review_id: String,
    pub scope: LandscapeScope,
    pub required_metric_ids: Vec<String>,
    pub alternatives: Vec<AlternativeInput>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReviewLevel {
    SchematicEvidenceReview,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReviewState {
    EvidenceGaps,
    SyntheticComparisonOnly,
    ReadyForHumanInspection,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum FindingSeverity {
    Info,
    Warning,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewFinding {
    pub code: String,
    pub severity: FindingSeverity,
    pub alternative_id: Option<String>,
    pub metric_id: Option<String>,
    pub related_sha256s: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MetricAssessment {
    pub metric_id: String,
    pub unit: Option<String>,
    pub state: MetricState,
    pub evidence_class: EvidenceClass,
    pub method_uri: Option<String>,
    pub uncertainty_summary: String,
    pub supporting_artifact_sha256s: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ComparabilityState {
    ComparableForHumanInspection,
    MissingEvidence,
    UnitMismatch,
    MethodMismatch,
    EvidenceClassMismatch,
    NotComputable,
    Conflicting,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MetricComparability {
    pub metric_id: String,
    pub state: ComparabilityState,
    pub units_by_alternative: BTreeMap<String, Option<String>>,
    pub explanation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AlternativeAssessment {
    pub alternative_id: String,
    pub project_snapshot_sha256: String,
    pub project_scope: LandscapeScope,
    pub geometry_status: String,
    pub coordinate_frame_kind: String,
    pub georeferenced: bool,
    pub verified_artifact_sha256s: Vec<String>,
    pub metrics: Vec<MetricAssessment>,
    pub missing_required_metric_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewReport {
    pub schema: String,
    pub schema_uri: String,
    pub review_id: String,
    pub scope: LandscapeScope,
    pub review_level: ReviewLevel,
    pub state: ReviewState,
    pub automated_ranking_allowed: bool,
    pub environmental_claims_allowed: bool,
    pub causal_claims_allowed: bool,
    pub alternatives: Vec<AlternativeAssessment>,
    pub metric_comparability: Vec<MetricComparability>,
    pub findings: Vec<ReviewFinding>,
    pub interpretation_limit: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReviewViolation {
    pub code: String,
    pub alternative_id: Option<String>,
    pub artifact_sha256: Option<String>,
    pub field: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug)]
struct ProjectMetadata {
    scope: LandscapeScope,
    geometry_status: String,
    frame_kind: String,
    georeferenced: bool,
    feature_count: usize,
}

#[derive(Clone, Debug)]
struct ValidatedAlternative {
    report: AlternativeAssessment,
    roles: BTreeMap<String, ArtifactRole>,
    scenario_purposes: BTreeMap<String, String>,
    bound_observation_batches: BTreeSet<String>,
    metric_evidence: BTreeMap<String, MetricEvidence>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Validate the references and produce a deterministic, read-only evidence
/// coverage report. Raw artifact bytes must be provided by the caller, keyed
/// by their claimed lowercase SHA-256 digests. This function performs no I/O.
pub fn validate_and_review(
    request: &ReviewRequest,
    payloads_by_sha256: &BTreeMap<String, Vec<u8>>,
) -> Result<ReviewReport, Vec<ReviewViolation>> {
    let mut violations = Vec::new();
    let mut findings = Vec::new();

    if request.schema != REQUEST_SCHEMA {
        push_violation(&mut violations, "unsupported-review-schema", None, None, Some("schema"), "The review request schema is not supported.");
    }
    if request.schema_uri != REQUEST_SCHEMA_URI {
        push_violation(&mut violations, "unsupported-review-schema-uri", None, None, Some("schema_uri"), "The review request schema URI is not supported.");
    }
    if !is_identifier(&request.review_id) {
        push_violation(&mut violations, "invalid-review-id", None, None, Some("review_id"), "review_id must be a non-empty portable identifier.");
    }
    if request.alternatives.len() < 2 || request.alternatives.len() > MAX_ALTERNATIVES {
        push_violation(&mut violations, "alternative-count-out-of-range", None, None, Some("alternatives"), "A review requires 2..=8 alternatives.");
    }
    if request.required_metric_ids.is_empty() || request.required_metric_ids.len() > MAX_REQUIRED_METRICS {
        push_violation(&mut violations, "required-metric-count-out-of-range", None, None, Some("required_metric_ids"), "The review requires 1..=64 metric IDs.");
    }

    let mut required_ids = BTreeSet::new();
    for metric_id in &request.required_metric_ids {
        if !is_metric_id(metric_id) {
            push_violation(&mut violations, "invalid-required-metric-id", None, None, Some("required_metric_ids"), "Metric IDs must be stable dotted identifiers.");
        }
        if !required_ids.insert(metric_id.clone()) {
            push_violation(&mut violations, "duplicate-required-metric-id", None, None, Some("required_metric_ids"), "Required metric IDs must be unique.");
        }
    }

    // Check the caller-provided payload map itself before any artifact is
    // interpreted. A map key is not trusted as proof of content identity.
    for (claimed_digest, bytes) in payloads_by_sha256 {
        if !is_sha256(claimed_digest) || sha256_hex(bytes) != *claimed_digest {
            push_violation(&mut violations, "payload-key-digest-mismatch", None, Some(claimed_digest), None, "A supplied payload key does not match the SHA-256 of its exact bytes.");
        }
    }

    let mut alternative_ids = BTreeSet::new();
    let mut project_digests = BTreeSet::new();
    let mut validated = Vec::new();
    for alternative in &request.alternatives {
        if !is_identifier(&alternative.alternative_id) {
            push_violation(&mut violations, "invalid-alternative-id", Some(&alternative.alternative_id), None, Some("alternative_id"), "Alternative IDs must be non-empty portable identifiers.");
        }
        if !alternative_ids.insert(alternative.alternative_id.clone()) {
            push_violation(&mut violations, "duplicate-alternative-id", Some(&alternative.alternative_id), None, Some("alternative_id"), "Alternative IDs must be unique.");
        }
        if !project_digests.insert(alternative.project_snapshot.sha256.clone()) {
            push_violation(&mut violations, "duplicate-project-snapshot", Some(&alternative.alternative_id), Some(&alternative.project_snapshot.sha256), Some("project_snapshot"), "Different alternatives cannot silently alias the same project snapshot digest.");
        }
        match validate_alternative(request, alternative, payloads_by_sha256, &mut violations, &mut findings) {
            Some(item) => validated.push(item),
            None => {}
        }
    }

    // Extra bytes are not silently absorbed into the proof context.
    let mut referenced_payloads = BTreeSet::new();
    for alternative in &request.alternatives {
        for reference in all_references(alternative) {
            referenced_payloads.insert(reference.sha256.clone());
        }
    }
    for digest in payloads_by_sha256.keys() {
        if !referenced_payloads.contains(digest) {
            push_violation(&mut violations, "unreferenced-payload", None, Some(digest), None, "The request supplied artifact bytes that no alternative references.");
        }
    }

    if !violations.is_empty() {
        violations.sort_by(|a, b| {
            (&a.code, &a.alternative_id, &a.artifact_sha256, &a.field, &a.detail)
                .cmp(&(&b.code, &b.alternative_id, &b.artifact_sha256, &b.field, &b.detail))
        });
        violations.dedup();
        return Err(violations);
    }

    // Normalize order so semantically identical requests produce byte-stable
    // report JSON regardless of input vector ordering.
    validated.sort_by(|a, b| a.report.alternative_id.cmp(&b.report.alternative_id));
    for item in &mut validated {
        item.report.metrics.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
        item.report.verified_artifact_sha256s.sort();
        item.report.verified_artifact_sha256s.dedup();
        item.report.missing_required_metric_ids.sort();
        item.report.missing_required_metric_ids.dedup();
    }

    let metric_comparability = build_metric_comparability(&validated, &request.required_metric_ids);
    let all_required_present = validated.iter().all(|item| item.report.missing_required_metric_ids.is_empty());
    let has_synthetic = validated.iter().flat_map(|item| item.report.metrics.iter()).any(|metric| metric.evidence_class == EvidenceClass::Synthetic);
    let state = if !all_required_present {
        ReviewState::EvidenceGaps
    } else if has_synthetic {
        ReviewState::SyntheticComparisonOnly
    } else {
        ReviewState::ReadyForHumanInspection
    };

    findings.sort_by(|a, b| {
        (&a.code, &a.alternative_id, &a.metric_id, &a.related_sha256s, &a.detail)
            .cmp(&(&b.code, &b.alternative_id, &b.metric_id, &b.related_sha256s, &b.detail))
    });
    findings.dedup();

    Ok(ReviewReport {
        schema: REPORT_SCHEMA.to_string(),
        schema_uri: REPORT_SCHEMA_URI.to_string(),
        review_id: request.review_id.clone(),
        scope: request.scope.clone(),
        review_level: ReviewLevel::SchematicEvidenceReview,
        state,
        // This review profile never ranks alternatives, even if coverage is complete.
        automated_ranking_allowed: false,
        environmental_claims_allowed: false,
        causal_claims_allowed: false,
        alternatives: validated.into_iter().map(|item| item.report).collect(),
        metric_comparability,
        findings,
        interpretation_limit: "Read-only evidence coverage only. Sol-Atlas v0 snapshots use schematic local drawing units. This report does not rank alternatives, establish environmental effects, establish causality, or prove consent/legal authority.".to_string(),
    })
}

fn validate_alternative(
    request: &ReviewRequest,
    alternative: &AlternativeInput,
    payloads: &BTreeMap<String, Vec<u8>>,
    violations: &mut Vec<ReviewViolation>,
    findings: &mut Vec<ReviewFinding>,
) -> Option<ValidatedAlternative> {
    let alt_id = alternative.alternative_id.as_str();
    if alternative.project_snapshot.role != ArtifactRole::ProjectSnapshot {
        push_violation(violations, "wrong-project-snapshot-role", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("project_snapshot.role"), "project_snapshot must have the project-snapshot role.");
        return None;
    }
    if alternative.project_snapshot.schema_uri != PROJECT_SCHEMA_URI {
        push_violation(violations, "unsupported-project-snapshot-schema-uri", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("project_snapshot.schema_uri"), "The v0 review crate only accepts the pinned Sol-Atlas snapshot schema.");
        return None;
    }
    let snapshot_bytes = verified_bytes(&alternative.project_snapshot, payloads, alt_id, violations)?;
    let snapshot = parse_object(snapshot_bytes, alt_id, &alternative.project_snapshot, violations)?;
    let metadata = validate_project_snapshot(&snapshot, request, alternative, violations)?;

    let mut roles = BTreeMap::new();
    let mut scenario_purposes = BTreeMap::new();
    let mut bound_observation_batches = BTreeSet::new();
    let mut verified_digests = BTreeSet::new();
    verified_digests.insert(alternative.project_snapshot.sha256.clone());
    roles.insert(alternative.project_snapshot.sha256.clone(), ArtifactRole::ProjectSnapshot);

    let mut evaluation_plan_id = None;
    if let Some(plan_ref) = &alternative.evaluation_plan {
        if plan_ref.role != ArtifactRole::EvaluationPlan || plan_ref.schema_uri != EVALUATION_PLAN_SCHEMA_URI {
            push_violation(violations, "unsupported-evaluation-plan-reference", Some(alt_id), Some(&plan_ref.sha256), Some("evaluation_plan"), "Evaluation plan reference must use the pinned v0 role and schema URI.");
        } else if let Some(bytes) = verified_bytes(plan_ref, payloads, alt_id, violations) {
            if let Some(value) = parse_object(bytes, alt_id, plan_ref, violations) {
                evaluation_plan_id = validate_evaluation_plan(&value, plan_ref, &alternative.project_snapshot.sha256, request.scope.as_str(), alt_id, violations);
                verified_digests.insert(plan_ref.sha256.clone());
                roles.insert(plan_ref.sha256.clone(), ArtifactRole::EvaluationPlan);
            }
        }
    }

    for scenario_ref in &alternative.scenario_manifests {
        if scenario_ref.role != ArtifactRole::ScenarioManifest || scenario_ref.schema_uri != SCENARIO_SCHEMA_URI {
            push_violation(violations, "unsupported-scenario-manifest-reference", Some(alt_id), Some(&scenario_ref.sha256), Some("scenario_manifests"), "Scenario manifest reference must use the pinned Symtropy v0 role and schema URI.");
            continue;
        }
        let Some(bytes) = verified_bytes(scenario_ref, payloads, alt_id, violations) else { continue };
        let Some(value) = parse_object(bytes, alt_id, scenario_ref, violations) else { continue };
        if let Some(purpose) = validate_scenario_manifest(&value, scenario_ref, &alternative.project_snapshot.sha256, &metadata, alternative.evaluation_plan.as_ref(), evaluation_plan_id.as_deref(), alt_id, violations) {
            scenario_purposes.insert(scenario_ref.sha256.clone(), purpose);
            verified_digests.insert(scenario_ref.sha256.clone());
            roles.insert(scenario_ref.sha256.clone(), ArtifactRole::ScenarioManifest);
        }
    }

    for result_ref in &alternative.scenario_results {
        if result_ref.role != ArtifactRole::ScenarioResult {
            push_violation(violations, "wrong-scenario-result-role", Some(alt_id), Some(&result_ref.sha256), Some("scenario_results"), "Each scenario result must use the scenario-result role.");
            continue;
        }
        let Some(bytes) = verified_bytes(result_ref, payloads, alt_id, violations) else { continue };
        let Some(value) = parse_object(bytes, alt_id, result_ref, violations) else { continue };
        if let Some(manifest_digest) = value.get("scenario_manifest_sha256").and_then(Value::as_str) {
            if !scenario_purposes.contains_key(manifest_digest) {
                push_violation(violations, "scenario-result-manifest-not-pinned", Some(alt_id), Some(&result_ref.sha256), Some("scenario_manifest_sha256"), "Scenario result must reference a scenario manifest verified for this alternative.");
                continue;
            }
            let evidence_class = value.get("evidence_class").and_then(Value::as_str).unwrap_or("");
            let purpose = scenario_purposes.get(manifest_digest).map(String::as_str).unwrap_or("");
            if evidence_class == "modelled" && (purpose == "schema-qualification" || purpose == "synthetic-benchmark") {
                push_violation(violations, "unqualified-modelled-result", Some(alt_id), Some(&result_ref.sha256), Some("evidence_class"), "The pinned Symtropy v0 manifest only permits schema qualification or synthetic benchmarking; it cannot qualify a physical model result.");
                continue;
            }
            if !matches!(evidence_class, "synthetic" | "modelled") {
                push_violation(violations, "unsupported-scenario-result-evidence-class", Some(alt_id), Some(&result_ref.sha256), Some("evidence_class"), "Scenario results must explicitly declare synthetic or modelled evidence.");
                continue;
            }
            if result_ref.schema_uri.is_empty() || !is_uri(&result_ref.schema_uri) {
                push_violation(violations, "invalid-scenario-result-schema-uri", Some(alt_id), Some(&result_ref.sha256), Some("schema_uri"), "Scenario result must declare an absolute schema URI.");
                continue;
            }
            if value.get("schema_uri").and_then(Value::as_str) != Some(result_ref.schema_uri.as_str()) {
                push_violation(violations, "scenario-result-schema-uri-mismatch", Some(alt_id), Some(&result_ref.sha256), Some("schema_uri"), "The result payload schema URI does not match its reference.");
                continue;
            }
            scenario_purposes.entry(result_ref.sha256.clone()).or_insert_with(|| evidence_class.to_string());
            verified_digests.insert(result_ref.sha256.clone());
            roles.insert(result_ref.sha256.clone(), ArtifactRole::ScenarioResult);
        } else {
            push_violation(violations, "scenario-result-missing-manifest-reference", Some(alt_id), Some(&result_ref.sha256), Some("scenario_manifest_sha256"), "Scenario result must bind to an exact verified manifest digest.");
        }
    }

    for observation_ref in &alternative.observation_batches {
        if observation_ref.role != ArtifactRole::ObservationBatch || observation_ref.schema_uri != OBSERVATION_BATCH_SCHEMA_URI {
            push_violation(violations, "unsupported-observation-batch-reference", Some(alt_id), Some(&observation_ref.sha256), Some("observation_batches"), "Observation batch reference must use the pinned Sol-Atlas v0 schema URI.");
            continue;
        }
        let Some(bytes) = verified_bytes(observation_ref, payloads, alt_id, violations) else { continue };
        let Some(value) = parse_object(bytes, alt_id, observation_ref, violations) else { continue };
        if validate_observation_batch(&value, observation_ref, &alternative.project_snapshot.sha256, alt_id, violations) {
            bound_observation_batches.insert(observation_ref.sha256.clone());
        }
        verified_digests.insert(observation_ref.sha256.clone());
        roles.insert(observation_ref.sha256.clone(), ArtifactRole::ObservationBatch);
    }

    for stewardship_ref in &alternative.stewardship_records {
        if stewardship_ref.role != ArtifactRole::StewardshipRecord || stewardship_ref.schema_uri != STEWARDSHIP_SCHEMA_URI {
            push_violation(violations, "unsupported-stewardship-reference", Some(alt_id), Some(&stewardship_ref.sha256), Some("stewardship_records"), "Stewardship record reference must use the pinned Mycelix profile URI.");
            continue;
        }
        let Some(bytes) = verified_bytes(stewardship_ref, payloads, alt_id, violations) else { continue };
        let Some(value) = parse_object(bytes, alt_id, stewardship_ref, violations) else { continue };
        if validate_stewardship_record(&value, stewardship_ref, &alternative.project_snapshot.sha256, &scenario_purposes, alt_id, violations) {
            verified_digests.insert(stewardship_ref.sha256.clone());
            roles.insert(stewardship_ref.sha256.clone(), ArtifactRole::StewardshipRecord);
        }
    }

    for source_ref in &alternative.source_datasets {
        if source_ref.role != ArtifactRole::SourceDataset {
            push_violation(violations, "wrong-source-dataset-role", Some(alt_id), Some(&source_ref.sha256), Some("source_datasets"), "Source data references must use the source-dataset role.");
            continue;
        }
        let Some(bytes) = verified_bytes(source_ref, payloads, alt_id, violations) else { continue };
        let Some(value) = parse_object(bytes, alt_id, source_ref, violations) else { continue };
        if validate_source_dataset(&value, source_ref, alt_id, violations) {
            verified_digests.insert(source_ref.sha256.clone());
            roles.insert(source_ref.sha256.clone(), ArtifactRole::SourceDataset);
        }
    }

    let mut metrics = BTreeMap::new();
    for metric in &alternative.metrics {
        if !request.required_metric_ids.contains(&metric.metric_id) {
            push_violation(violations, "unrequested-metric", Some(alt_id), None, Some("metrics.metric_id"), "Metric is not listed in required_metric_ids; version the review request before adding a metric.");
            continue;
        }
        if !is_metric_id(&metric.metric_id) || metric.unit.trim().is_empty() || metric.uncertainty_summary.trim().is_empty() {
            push_violation(violations, "invalid-metric-evidence", Some(alt_id), None, Some("metrics"), "Metric ID, unit, and uncertainty summary must be explicit.");
            continue;
        }
        if metrics.insert(metric.metric_id.clone(), metric.clone()).is_some() {
            push_violation(violations, "duplicate-metric-evidence", Some(alt_id), None, Some("metrics.metric_id"), "Each required metric can appear at most once per alternative.");
            continue;
        }
        let supports: BTreeSet<String> = metric.supporting_artifact_sha256s.iter().cloned().collect();
        if supports.len() != metric.supporting_artifact_sha256s.len() {
            push_violation(violations, "duplicate-metric-support-reference", Some(alt_id), None, Some("metrics.supporting_artifact_sha256s"), "Supporting artifact digests must be unique.");
        }
        for support in &supports {
            if !roles.contains_key(support) {
                push_violation(violations, "metric-support-not-verified", Some(alt_id), Some(support), Some("metrics.supporting_artifact_sha256s"), "Every supporting digest must reference an artifact verified in this alternative.");
            }
        }
        validate_metric_evidence(metric, alt_id, &roles, &scenario_purposes, &bound_observation_batches, violations);
        if matches!(metric.state, MetricState::Missing | MetricState::NotComputable | MetricState::Conflicting) {
            findings.push(ReviewFinding {
                code: match metric.state {
                    MetricState::Missing => "metric-evidence-missing".to_string(),
                    MetricState::NotComputable => "metric-not-computable".to_string(),
                    MetricState::Conflicting => "metric-evidence-conflicting".to_string(),
                    MetricState::Available => unreachable!(),
                },
                severity: FindingSeverity::Warning,
                alternative_id: Some(alt_id.to_string()),
                metric_id: Some(metric.metric_id.clone()),
                related_sha256s: metric.supporting_artifact_sha256s.clone(),
                detail: metric.uncertainty_summary.clone(),
            });
        }
        if metric.evidence_class == EvidenceClass::Synthetic {
            findings.push(ReviewFinding {
                code: "synthetic-evidence-claim-ceiling".to_string(),
                severity: FindingSeverity::Info,
                alternative_id: Some(alt_id.to_string()),
                metric_id: Some(metric.metric_id.clone()),
                related_sha256s: metric.supporting_artifact_sha256s.clone(),
                detail: "This evidence remains synthetic and cannot support a field-effect or environmental-impact claim.".to_string(),
            });
        }
    }

    let mut assessments = Vec::new();
    let mut missing_metrics = Vec::new();
    for required_id in &request.required_metric_ids {
        match metrics.get(required_id) {
            Some(metric) => assessments.push(metric_assessment(metric)),
            None => {
                missing_metrics.push(required_id.clone());
                assessments.push(MetricAssessment {
                    metric_id: required_id.clone(),
                    unit: None,
                    state: MetricState::Missing,
                    evidence_class: EvidenceClass::Missing,
                    method_uri: None,
                    uncertainty_summary: "No evidence record was supplied for this required metric; no value is imputed.".to_string(),
                    supporting_artifact_sha256s: Vec::new(),
                });
                findings.push(ReviewFinding {
                    code: "required-metric-not-provided".to_string(),
                    severity: FindingSeverity::Warning,
                    alternative_id: Some(alt_id.to_string()),
                    metric_id: Some(required_id.clone()),
                    related_sha256s: vec![alternative.project_snapshot.sha256.clone()],
                    detail: "The metric was requested but no evidence record was supplied.".to_string(),
                });
            }
        }
    }
    assessments.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
    missing_metrics.sort();
    let report = AlternativeAssessment {
        alternative_id: alt_id.to_string(),
        project_snapshot_sha256: alternative.project_snapshot.sha256.clone(),
        project_scope: metadata.scope,
        geometry_status: metadata.geometry_status,
        coordinate_frame_kind: metadata.frame_kind,
        georeferenced: metadata.georeferenced,
        verified_artifact_sha256s: verified_digests.into_iter().collect(),
        metrics: assessments,
        missing_required_metric_ids: missing_metrics,
    };
    Some(ValidatedAlternative {
        report,
        roles,
        scenario_purposes,
        bound_observation_batches,
        metric_evidence: metrics,
    })
}

fn validate_project_snapshot(
    value: &Value,
    request: &ReviewRequest,
    alternative: &AlternativeInput,
    violations: &mut Vec<ReviewViolation>,
) -> ProjectMetadata {
    let alt_id = alternative.alternative_id.as_str();
    if value.get("schema").and_then(Value::as_str) != Some(PROJECT_SCHEMA)
        || value.get("schema_uri").and_then(Value::as_str) != Some(PROJECT_SCHEMA_URI)
    {
        push_violation(violations, "project-snapshot-schema-mismatch", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("schema"), "Payload does not match the pinned Sol-Atlas v0 snapshot schema.");
    }
    let scope_str = value.get("scope").and_then(Value::as_str).unwrap_or("");
    let scope = parse_scope(scope_str).unwrap_or_else(|| request.scope.clone());
    if scope_str != request.scope.as_str() {
        push_violation(violations, "project-scope-mismatch", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("scope"), "Project snapshot scope must equal the review request scope.");
    }
    let frame_kind = value.pointer("/coordinate_frame/kind").and_then(Value::as_str).unwrap_or("").to_string();
    let georeferenced = value.pointer("/coordinate_frame/georeferenced").and_then(Value::as_bool).unwrap_or(true);
    let geometry_status = value.get("geometry_status").and_then(Value::as_str).unwrap_or("").to_string();
    if frame_kind != LOCAL_FRAME || georeferenced {
        push_violation(violations, "unsupported-project-coordinate-frame", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("coordinate_frame"), "The v0 review profile only accepts explicit local drawing units with georeferenced=false.");
    }
    if !matches!(geometry_status.as_str(), "schematic-design-intent" | "synthetic-benchmark") {
        push_violation(violations, "unsupported-project-geometry-status", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("geometry_status"), "The project geometry status is not supported by the v0 review contract.");
    }
    let features = value.get("features").and_then(Value::as_array);
    if features.is_none() {
        push_violation(violations, "project-feature-list-missing", Some(alt_id), Some(&alternative.project_snapshot.sha256), Some("features"), "Project snapshot must contain a feature array.");
    }
    ProjectMetadata {
        scope,
        geometry_status,
        frame_kind,
        georeferenced,
        feature_count: features.map(Vec::len).unwrap_or_default(),
    }
}

fn validate_evaluation_plan(
    value: &Value,
    reference: &ArtifactRef,
    project_digest: &str,
    scope: &str,
    alternative_id: &str,
    violations: &mut Vec<ReviewViolation>,
) -> Option<String> {
    if value.get("schema").and_then(Value::as_str) != Some(EVALUATION_PLAN_SCHEMA)
        || value.get("schema_uri").and_then(Value::as_str) != Some(EVALUATION_PLAN_SCHEMA_URI)
        || reference.schema_uri != EVALUATION_PLAN_SCHEMA_URI
    {
        push_violation(violations, "evaluation-plan-schema-mismatch", Some(alternative_id), Some(&reference.sha256), Some("schema"), "Evaluation plan does not match the pinned Sol-Atlas evaluation-plan schema.");
    }
    if value.get("project_snapshot_sha256").and_then(Value::as_str) != Some(project_digest) {
        push_violation(violations, "evaluation-project-digest-mismatch", Some(alternative_id), Some(&reference.sha256), Some("project_snapshot_sha256"), "Evaluation plan must pin this alternative's exact project snapshot digest.");
    }
    if value.get("project_scope").and_then(Value::as_str) != Some(scope) {
        push_violation(violations, "evaluation-scope-mismatch", Some(alternative_id), Some(&reference.sha256), Some("project_scope"), "Evaluation plan scope must match the project snapshot and review scope.");
    }
    value.get("evaluation_id").and_then(Value::as_str).map(str::to_string)
}

fn validate_scenario_manifest(
    value: &Value,
    reference: &ArtifactRef,
    project_digest: &str,
    project: &ProjectMetadata,
    evaluation_plan: Option<&ArtifactRef>,
    evaluation_plan_id: Option<&str>,
    alternative_id: &str,
    violations: &mut Vec<ReviewViolation>,
) -> Option<String> {
    if value.get("schema").and_then(Value::as_str) != Some(SCENARIO_SCHEMA)
        || reference.schema_uri != SCENARIO_SCHEMA_URI
    {
        push_violation(violations, "scenario-manifest-schema-mismatch", Some(alternative_id), Some(&reference.sha256), Some("schema"), "Scenario manifest does not use the pinned Symtropy scenario schema.");
    }
    let design = value.get("design_snapshot");
    if design.and_then(|v| v.get("sha256")).and_then(Value::as_str) != Some(project_digest) {
        push_violation(violations, "scenario-project-digest-mismatch", Some(alternative_id), Some(&reference.sha256), Some("design_snapshot.sha256"), "Scenario manifest must pin the exact project snapshot digest.");
    }
    if design.and_then(|v| v.get("schema_uri")).and_then(Value::as_str) != Some(PROJECT_SCHEMA_URI)
        || design.and_then(|v| v.get("scope")).and_then(Value::as_str) != Some(project.scope.as_str())
        || design.and_then(|v| v.get("coordinate_frame_kind")).and_then(Value::as_str) != Some(project.frame_kind.as_str())
        || design.and_then(|v| v.get("georeferenced")).and_then(Value::as_bool) != Some(project.georeferenced)
        || design.and_then(|v| v.get("geometry_status")).and_then(Value::as_str) != Some(project.geometry_status.as_str())
    {
        push_violation(violations, "scenario-project-metadata-mismatch", Some(alternative_id), Some(&reference.sha256), Some("design_snapshot"), "Scenario manifest project metadata must agree with the exact Sol-Atlas snapshot.");
    }
    let purpose = value.get("purpose").and_then(Value::as_str).unwrap_or("").to_string();
    if !matches!(purpose.as_str(), "schema-qualification" | "synthetic-benchmark") {
        push_violation(violations, "unsupported-scenario-purpose", Some(alternative_id), Some(&reference.sha256), Some("purpose"), "The current Symtropy v0 contract permits only schema qualification and synthetic benchmarking.");
    }
    if purpose == "synthetic-benchmark" && project.geometry_status != "synthetic-benchmark" {
        push_violation(violations, "synthetic-purpose-project-status-mismatch", Some(alternative_id), Some(&reference.sha256), Some("purpose"), "A synthetic-benchmark scenario must use a project snapshot explicitly marked synthetic-benchmark.");
    }
    if design.and_then(|v| v.get("feature_count")).and_then(Value::as_u64) != Some(project.feature_count as u64) {
        push_violation(violations, "scenario-feature-count-mismatch", Some(alternative_id), Some(&reference.sha256), Some("design_snapshot.feature_count"), "Scenario manifest feature count must equal the pinned project snapshot feature count.");
    }
    match (evaluation_plan, value.get("evaluation_plan")) {
        (Some(plan_ref), Some(manifest_plan)) => {
            let manifest_digest = manifest_plan.get("sha256").and_then(Value::as_str);
            let manifest_id = manifest_plan.get("evaluation_id").and_then(Value::as_str);
            if manifest_digest != Some(plan_ref.sha256.as_str())
                || manifest_id.is_none()
                || evaluation_plan_id.is_some_and(|expected_id| manifest_id != Some(expected_id))
                || manifest_plan.get("schema_uri").and_then(Value::as_str) != Some(EVALUATION_PLAN_SCHEMA_URI)
            {
                push_violation(violations, "scenario-evaluation-plan-mismatch", Some(alternative_id), Some(&reference.sha256), Some("evaluation_plan"), "Scenario's optional evaluation-plan reference must match the exact pinned plan digest, schema URI, and evaluation ID.");
            }
        }
        (Some(_), None) => {
            push_violation(violations, "scenario-missing-evaluation-plan-link", Some(alternative_id), Some(&reference.sha256), Some("evaluation_plan"), "This review pinned an evaluation plan; scenario manifests in the review must reference that same plan.");
        }
        (None, Some(_)) => {
            push_violation(violations, "scenario-evaluation-plan-bytes-not-pinned", Some(alternative_id), Some(&reference.sha256), Some("evaluation_plan"), "Scenario manifest references an evaluation plan but the review did not pin its exact bytes.");
        }
        (None, None) => {}
    }
    Some(purpose)
}

fn validate_observation_batch(
    value: &Value,
    reference: &ArtifactRef,
    project_digest: &str,
    alternative_id: &str,
    violations: &mut Vec<ReviewViolation>,
) -> bool {
    if value.get("schema").and_then(Value::as_str) != Some(OBSERVATION_BATCH_SCHEMA)
        || value.get("schema_uri").and_then(Value::as_str) != Some(OBSERVATION_BATCH_SCHEMA_URI)
        || reference.schema_uri != OBSERVATION_BATCH_SCHEMA_URI
    {
        push_violation(violations, "observation-batch-schema-mismatch", Some(alternative_id), Some(&reference.sha256), Some("schema"), "Observation batch must use the pinned Sol-Atlas observation batch schema.");
    }
    let Some(observations) = value.get("observations").and_then(Value::as_array) else {
        push_violation(violations, "observation-batch-records-missing", Some(alternative_id), Some(&reference.sha256), Some("observations"), "Observation batch must contain an observations array.");
        return false;
    };
    let mut bound = false;
    for observation in observations {
        let bound_snapshot = observation.pointer("/project_ref/project_snapshot_sha256").and_then(Value::as_str);
        match bound_snapshot {
            Some(digest) if digest == project_digest => bound = true,
            Some(_) => push_violation(violations, "observation-project-digest-mismatch", Some(alternative_id), Some(&reference.sha256), Some("observations.project_ref.project_snapshot_sha256"), "An observation attached to a different project revision cannot support this alternative."),
            None => {}
        }
    }
    bound
}

fn validate_stewardship_record(
    value: &Value,
    reference: &ArtifactRef,
    project_digest: &str,
    scenario_purposes: &BTreeMap<String, String>,
    alternative_id: &str,
    violations: &mut Vec<ReviewViolation>,
) -> bool {
    if value.get("schema").and_then(Value::as_str) != Some(STEWARDSHIP_SCHEMA)
        || reference.schema_uri != STEWARDSHIP_SCHEMA_URI
    {
        push_violation(violations, "stewardship-schema-mismatch", Some(alternative_id), Some(&reference.sha256), Some("schema"), "Stewardship record must use the pinned Mycelix profile v0 schema URI.");
        return false;
    }
    if value.pointer("/project_snapshot/sha256").and_then(Value::as_str) != Some(project_digest) {
        push_violation(violations, "stewardship-project-digest-mismatch", Some(alternative_id), Some(&reference.sha256), Some("project_snapshot.sha256"), "Stewardship record must reference the exact project revision being reviewed.");
        return false;
    }
    if let Some(manifest_digest) = value.pointer("/scenario_reference/manifest_sha256").and_then(Value::as_str) {
        if !scenario_purposes.contains_key(manifest_digest) {
            push_violation(violations, "stewardship-scenario-not-pinned", Some(alternative_id), Some(&reference.sha256), Some("scenario_reference.manifest_sha256"), "Stewardship scenario reference must point to a verified scenario manifest in this alternative.");
            return false;
        }
    }
    // This is only a reference check. Mycelix remains authoritative for access,
    // actor authority, consent, withdrawal, and publication policy semantics.
    true
}

fn validate_source_dataset(
    value: &Value,
    reference: &ArtifactRef,
    alternative_id: &str,
    violations: &mut Vec<ReviewViolation>,
) -> bool {
    if let Some(schema_uri) = value.get("schema_uri").and_then(Value::as_str) {
        if schema_uri != reference.schema_uri {
            push_violation(violations, "source-schema-uri-mismatch", Some(alternative_id), Some(&reference.sha256), Some("schema_uri"), "Source dataset's embedded schema URI must match its reference.");
            return false;
        }
    }
    if value.get("ingestion_state").and_then(Value::as_str) != Some("ingested") {
        push_violation(violations, "source-not-ingested", Some(alternative_id), Some(&reference.sha256), Some("ingestion_state"), "A catalog candidate or metadata-only entry is not a verified ingested dataset and cannot support a metric.");
        return false;
    }
    let has_identity = value.get("source_id").and_then(Value::as_str).is_some()
        || value.get("dataset_id").and_then(Value::as_str).is_some();
    if !has_identity {
        push_violation(violations, "source-identity-missing", Some(alternative_id), Some(&reference.sha256), Some("source_id"), "An ingested dataset must carry a stable source or dataset identifier.");
        return false;
    }
    true
}

fn validate_metric_evidence(
    metric: &MetricEvidence,
    alternative_id: &str,
    roles: &BTreeMap<String, ArtifactRole>,
    scenario_purposes: &BTreeMap<String, String>,
    bound_observation_batches: &BTreeSet<String>,
    violations: &mut Vec<ReviewViolation>,
) {
    if metric.state == MetricState::Available {
        if metric.evidence_class == EvidenceClass::Missing {
            push_violation(violations, "available-metric-marked-missing", Some(alternative_id), None, Some("metrics.evidence_class"), "An available metric cannot be labeled missing.");
        }
        if metric.method_uri.as_deref().map(str::trim).unwrap_or("").is_empty() || metric.uncertainty_summary.trim().is_empty() {
            push_violation(violations, "metric-method-or-uncertainty-missing", Some(alternative_id), None, Some("metrics"), "Available metric evidence must declare its method and uncertainty/limitation summary.");
        }
        if metric.supporting_artifact_sha256s.is_empty() {
            push_violation(violations, "metric-support-missing", Some(alternative_id), None, Some("metrics.supporting_artifact_sha256s"), "An available metric must reference at least one verified supporting artifact.");
        }
    }
    if metric.evidence_class == EvidenceClass::Missing && metric.state == MetricState::Available {
        push_violation(violations, "missing-evidence-cannot-be-available", Some(alternative_id), None, Some("metrics.state"), "Missing evidence cannot be presented as available.");
    }
    if metric.evidence_class == EvidenceClass::Observed {
        let has_bound_observation = metric.supporting_artifact_sha256s.iter().any(|digest| {
            roles.get(digest) == Some(&ArtifactRole::ObservationBatch) && bound_observation_batches.contains(digest)
        });
        if metric.state == MetricState::Available && !has_bound_observation {
            push_violation(violations, "observed-metric-without-project-bound-observation", Some(alternative_id), None, Some("metrics.supporting_artifact_sha256s"), "Observed evidence must reference a verified observation batch containing an observation bound to this exact snapshot digest.");
        }
    }
    if metric.evidence_class == EvidenceClass::Curated
        && metric.state == MetricState::Available
        && !metric.supporting_artifact_sha256s.iter().any(|digest| roles.get(digest) == Some(&ArtifactRole::SourceDataset))
    {
        push_violation(violations, "curated-metric-without-ingested-source", Some(alternative_id), None, Some("metrics.supporting_artifact_sha256s"), "Curated evidence must reference a dataset manifest whose ingestion_state is ingested.");
    }
    if metric.evidence_class == EvidenceClass::Modelled && metric.state == MetricState::Available {
        let result_digest = metric.supporting_artifact_sha256s.iter().find(|digest| roles.get(*digest) == Some(&ArtifactRole::ScenarioResult));
        match result_digest {
            Some(digest) => {
                let manifest_digest = metric.supporting_artifact_sha256s.iter().find(|item| roles.get(*item) == Some(&ArtifactRole::ScenarioManifest));
                let supported_purpose = manifest_digest.and_then(|manifest| scenario_purposes.get(manifest));
                if supported_purpose.is_none_or(|purpose| purpose == "schema-qualification" || purpose == "synthetic-benchmark") {
                    push_violation(violations, "modelled-metric-exceeds-pinned-scenario-claim-ceiling", Some(alternative_id), Some(digest), Some("metrics.evidence_class"), "Current Symtropy v0 scenarios permit schema qualification or synthetic benchmarking only; modelled landscape outcomes are not eligible.");
                }
            }
            None => push_violation(violations, "modelled-metric-without-result-artifact", Some(alternative_id), None, Some("metrics.supporting_artifact_sha256s"), "Modelled evidence must point to a verified scenario-result artifact."),
        }
    }
    if metric.evidence_class == EvidenceClass::Synthetic
        && metric.state == MetricState::Available
        && !metric.supporting_artifact_sha256s.iter().any(|digest| {
            (roles.get(digest) == Some(&ArtifactRole::ScenarioManifest)
                && scenario_purposes.get(digest).is_some_and(|purpose| purpose == "synthetic-benchmark"))
                || roles.get(digest) == Some(&ArtifactRole::ObservationBatch)
        })
    {
        push_violation(violations, "synthetic-metric-without-synthetic-source", Some(alternative_id), None, Some("metrics.supporting_artifact_sha256s"), "Synthetic evidence must cite a pinned synthetic scenario or synthetic observation batch.");
    }
}

fn build_metric_comparability(
    alternatives: &[ValidatedAlternative],
    required_metric_ids: &[String],
) -> Vec<MetricComparability> {
    let mut results = Vec::new();
    for metric_id in required_metric_ids {
        let entries: Vec<Option<&MetricEvidence>> = alternatives
            .iter()
            .map(|alternative| alternative.metric_evidence.get(metric_id))
            .collect();
        let mut units_by_alternative = BTreeMap::new();
        for alternative in alternatives {
            let unit = alternative.metric_evidence.get(metric_id).map(|metric| metric.unit.clone());
            units_by_alternative.insert(alternative.report.alternative_id.clone(), unit);
        }
        let state = if entries.iter().any(Option::is_none) {
            ComparabilityState::MissingEvidence
        } else {
            let metrics: Vec<&MetricEvidence> = entries.iter().filter_map(|entry| *entry).collect();
            if metrics.iter().any(|metric| metric.state == MetricState::Conflicting) {
                ComparabilityState::Conflicting
            } else if metrics.iter().any(|metric| metric.state != MetricState::Available || metric.evidence_class == EvidenceClass::Missing) {
                ComparabilityState::NotComputable
            } else if metrics.iter().any(|metric| metric.unit != metrics[0].unit) {
                ComparabilityState::UnitMismatch
            } else if metrics.iter().any(|metric| metric.method_uri != metrics[0].method_uri) {
                ComparabilityState::MethodMismatch
            } else if metrics.iter().any(|metric| metric.evidence_class != metrics[0].evidence_class) {
                ComparabilityState::EvidenceClassMismatch
            } else if matches!(metrics[0].evidence_class, EvidenceClass::Synthetic | EvidenceClass::Assumed | EvidenceClass::Modelled) {
                ComparabilityState::NotComputable
            } else {
                ComparabilityState::ComparableForHumanInspection
            }
        };
        let explanation = match state {
            ComparabilityState::ComparableForHumanInspection => "The metric uses the same unit, method URI, evidence class and available-state declaration across alternatives. This permits human inspection only, not ranking or causal inference.".to_string(),
            ComparabilityState::MissingEvidence => "At least one alternative has no record for this required metric.".to_string(),
            ComparabilityState::UnitMismatch => "Units differ across alternatives; do not compare or normalize implicitly.".to_string(),
            ComparabilityState::MethodMismatch => "Method URIs differ across alternatives; method equivalence must be reviewed explicitly.".to_string(),
            ComparabilityState::EvidenceClassMismatch => "Evidence classes differ across alternatives; observed, curated, synthetic, modelled, assumed and missing evidence are not interchangeable.".to_string(),
            ComparabilityState::Conflicting => "At least one alternative has conflicting evidence for this metric.".to_string(),
            ComparabilityState::NotComputable => "At least one alternative is unavailable, missing, assumed, synthetic or modelled under the current schematic claim ceiling.".to_string(),
        };
        results.push(MetricComparability { metric_id: metric_id.clone(), state, units_by_alternative, explanation });
    }
    results
}

fn metric_assessment(metric: &MetricEvidence) -> MetricAssessment {
    let mut supporting = metric.supporting_artifact_sha256s.clone();
    supporting.sort();
    supporting.dedup();
    MetricAssessment {
        metric_id: metric.metric_id.clone(),
        unit: Some(metric.unit.clone()),
        state: metric.state.clone(),
        evidence_class: metric.evidence_class.clone(),
        method_uri: metric.method_uri.clone(),
        uncertainty_summary: metric.uncertainty_summary.clone(),
        supporting_artifact_sha256s: supporting,
    }
}

fn all_references(alternative: &AlternativeInput) -> Vec<&ArtifactRef> {
    let mut refs = vec![&alternative.project_snapshot];
    if let Some(plan) = &alternative.evaluation_plan { refs.push(plan); }
    refs.extend(alternative.scenario_manifests.iter());
    refs.extend(alternative.scenario_results.iter());
    refs.extend(alternative.observation_batches.iter());
    refs.extend(alternative.stewardship_records.iter());
    refs.extend(alternative.source_datasets.iter());
    refs
}

fn verified_bytes<'a>(
    reference: &ArtifactRef,
    payloads: &'a BTreeMap<String, Vec<u8>>,
    alternative_id: &str,
    violations: &mut Vec<ReviewViolation>,
) -> Option<&'a [u8]> {
    if !is_sha256(&reference.sha256) {
        push_violation(violations, "invalid-artifact-digest", Some(alternative_id), Some(&reference.sha256), Some("sha256"), "Artifact digest must be 64 lowercase hexadecimal characters.");
        return None;
    }
    if !is_uri(&reference.schema_uri) {
        push_violation(violations, "invalid-artifact-schema-uri", Some(alternative_id), Some(&reference.sha256), Some("schema_uri"), "Artifact schema URI must be an absolute HTTPS or URN URI.");
        return None;
    }
    let Some(bytes) = payloads.get(&reference.sha256) else {
        push_violation(violations, "artifact-bytes-missing", Some(alternative_id), Some(&reference.sha256), None, "The exact bytes for this digest were not supplied to the read-only review call.");
        return None;
    };
    if sha256_hex(bytes) != reference.sha256 {
        push_violation(violations, "artifact-digest-mismatch", Some(alternative_id), Some(&reference.sha256), None, "SHA-256 of supplied bytes does not match the artifact reference.");
        return None;
    }
    Some(bytes.as_slice())
}

fn parse_object<'a>(
    bytes: &'a [u8],
    alternative_id: &str,
    reference: &ArtifactRef,
    violations: &mut Vec<ReviewViolation>,
) -> Option<Value> {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(value) if value.is_object() => Some(value),
        Ok(_) => {
            push_violation(violations, "artifact-not-json-object", Some(alternative_id), Some(&reference.sha256), None, "The pinned artifact must be a JSON object for this v0 review profile.");
            None
        }
        Err(error) => {
            push_violation(violations, "artifact-json-invalid", Some(alternative_id), Some(&reference.sha256), None, &format!("Pinned artifact is not valid JSON: {error}"));
            None
        }
    }
}

fn parse_scope(value: &str) -> Option<LandscapeScope> {
    match value {
        "site-parcel" => Some(LandscapeScope::SiteParcel),
        "district" => Some(LandscapeScope::District),
        "watershed" => Some(LandscapeScope::Watershed),
        "bioregion" => Some(LandscapeScope::Bioregion),
        _ => None,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_uri(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("urn:")
}

fn is_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 100
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
}

fn is_metric_id(value: &str) -> bool {
    value.len() <= 200
        && value.contains('.')
        && value.split('.').all(|part| {
            !part.is_empty()
                && part.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
        })
}

fn push_violation(
    violations: &mut Vec<ReviewViolation>,
    code: &str,
    alternative_id: Option<&str>,
    artifact_sha256: Option<&str>,
    field: Option<&str>,
    detail: &str,
) {
    violations.push(ReviewViolation {
        code: code.to_string(),
        alternative_id: alternative_id.map(str::to_string),
        artifact_sha256: artifact_sha256.map(str::to_string),
        field: field.map(str::to_string),
        detail: detail.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn project_bytes() -> Vec<u8> {
        include_bytes!("../fixtures/seed-landscape-v0.json").to_vec()
    }

    fn artifact_ref(role: ArtifactRole, bytes: &[u8], schema_uri: &str) -> ArtifactRef {
        ArtifactRef { role, sha256: sha256_hex(bytes), schema_uri: schema_uri.to_string() }
    }

    fn request_with_two_schematic_alternatives() -> (ReviewRequest, BTreeMap<String, Vec<u8>>) {
        let mut payloads = BTreeMap::new();
        let project_a = project_bytes();
        let mut snapshot_b: Value = serde_json::from_slice(&project_a).unwrap();
        snapshot_b["project_name"] = json!("Alternative B (schematic)");
        snapshot_b["features"][0]["x_du"] = json!(181);
        let project_b = serde_json::to_vec(&snapshot_b).unwrap();
        let ref_a = artifact_ref(ArtifactRole::ProjectSnapshot, &project_a, PROJECT_SCHEMA_URI);
        let ref_b = artifact_ref(ArtifactRole::ProjectSnapshot, &project_b, PROJECT_SCHEMA_URI);
        payloads.insert(ref_a.sha256.clone(), project_a);
        payloads.insert(ref_b.sha256.clone(), project_b);
        let alt = |alternative_id: &str, reference: ArtifactRef| AlternativeInput {
            alternative_id: alternative_id.to_string(),
            project_snapshot: reference,
            evaluation_plan: None,
            scenario_manifests: Vec::new(),
            scenario_results: Vec::new(),
            observation_batches: Vec::new(),
            stewardship_records: Vec::new(),
            source_datasets: Vec::new(),
            metrics: Vec::new(),
        };
        (
            ReviewRequest {
                schema: REQUEST_SCHEMA.to_string(),
                schema_uri: REQUEST_SCHEMA_URI.to_string(),
                review_id: "landscape-review-fixture-v0".to_string(),
                scope: LandscapeScope::SiteParcel,
                required_metric_ids: vec!["heat.shade_fraction".to_string(), "water.runoff_volume".to_string()],
                alternatives: vec![alt("option-b", ref_b), alt("option-a", ref_a)],
            },
            payloads,
        )
    }

    #[test]
    fn report_never_ranks_schematic_alternatives_or_allows_environmental_claims() {
        let (request, payloads) = request_with_two_schematic_alternatives();
        let report = validate_and_review(&request, &payloads).unwrap();
        assert_eq!(report.review_level, ReviewLevel::SchematicEvidenceReview);
        assert!(!report.automated_ranking_allowed);
        assert!(!report.environmental_claims_allowed);
        assert!(!report.causal_claims_allowed);
        assert_eq!(report.alternatives.len(), 2);
        assert!(report.alternatives.iter().all(|alt| {
            alt.coordinate_frame_kind == LOCAL_FRAME && !alt.georeferenced
        }));
        assert_eq!(report.state, ReviewState::EvidenceGaps);
    }

    #[test]
    fn raw_snapshot_digest_binds_exact_bytes() {
        let (request, payloads) = request_with_two_schematic_alternatives();
        let mut mutated_payloads = payloads.clone();
        let reference = &request.alternatives[0].project_snapshot;
        let bytes = mutated_payloads.get_mut(&reference.sha256).unwrap();
        bytes.push(b' ');
        let errors = validate_and_review(&request, &mutated_payloads).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "payload-key-digest-mismatch"));
    }

    #[test]
    fn absent_metric_is_reported_as_unknown_not_zero() {
        let (request, payloads) = request_with_two_schematic_alternatives();
        let report = validate_and_review(&request, &payloads).unwrap();
        assert!(report.alternatives.iter().all(|alt| {
            alt.metrics.iter().all(|metric| metric.state == MetricState::Missing && metric.unit.is_none())
        }));
        assert!(report.findings.iter().any(|finding| finding.code == "required-metric-not-provided"));
    }

    #[test]
    fn changed_scope_is_rejected() {
        let (mut request, payloads) = request_with_two_schematic_alternatives();
        request.scope = LandscapeScope::Watershed;
        let errors = validate_and_review(&request, &payloads).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "project-scope-mismatch"));
    }

    #[test]
    fn unknown_payload_bytes_are_not_silently_absorbed() {
        let (request, mut payloads) = request_with_two_schematic_alternatives();
        payloads.insert(sha256_hex(b"unreferenced"), b"unreferenced".to_vec());
        let errors = validate_and_review(&request, &payloads).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "unreferenced-payload"));
    }

    #[test]
    fn report_serialization_is_deterministic() {
        let (mut request, payloads) = request_with_two_schematic_alternatives();
        let first = validate_and_review(&request, &payloads).unwrap();
        request.alternatives.reverse();
        let second = validate_and_review(&request, &payloads).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn metric_unit_mismatch_never_becomes_comparable() {
        let (mut request, mut payloads) = request_with_two_schematic_alternatives();
        let alt_ids: Vec<String> = request.alternatives.iter().map(|alt| alt.alternative_id.clone()).collect();
        for (index, alt) in request.alternatives.iter_mut().enumerate() {
            let metric = MetricEvidence {
                metric_id: "heat.shade_fraction".to_string(),
                unit: if index == 0 { "percent".to_string() } else { "ratio".to_string() },
                state: MetricState::Available,
                evidence_class: EvidenceClass::Observed,
                method_uri: Some("https://example.invalid/method/shade-v1".to_string()),
                uncertainty_summary: "Synthetic test record; not an actual observation.".to_string(),
                supporting_artifact_sha256s: vec![alt.project_snapshot.sha256.clone()],
            };
            alt.metrics.push(metric);
            // Observed evidence cannot be sourced from a project snapshot.
            // Re-labeling it synthetic remains non-comparable for physical claims.
            alt.metrics[0].evidence_class = EvidenceClass::Synthetic;
        }
        let report = validate_and_review(&request, &payloads).unwrap();
        let cmp = report.metric_comparability.iter().find(|item| item.metric_id == "heat.shade_fraction").unwrap();
        assert_eq!(cmp.state, ComparabilityState::UnitMismatch);
        assert!(!report.automated_ranking_allowed);
        let _ = &mut payloads;
        let _ = alt_ids;
    }
}
