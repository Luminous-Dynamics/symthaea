//! Evidence-first component audit for agent benchmark validity.
//!
//! The manifest binds User/Task, Environment, Ground Truth and Evaluation
//! artifacts and maps explicit requirements to evaluator checks. The auditor
//! validates declared artifact/interface/coverage consistency; it does not
//! semantically prove prompt clarity, environment realism, oracle correctness
//! or grader fairness. Confirmatory qualification remains hard-false.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const COMPONENT_MANIFEST_SCHEMA: &str =
    "symthaea.benchmark_component_manifest.v1";
pub const COMPONENT_AUDIT_SCHEMA: &str =
    "symthaea.benchmark_component_audit.v1";
pub const COMPONENT_REVIEW_LEDGER_SCHEMA: &str =
    "symthaea.benchmark_component_review_ledger.v1";
pub const COMPONENT_AUDIT_MAX_REQUIREMENTS: usize = 100_000;
pub const COMPONENT_AUDIT_MAX_CHECKS: usize = 100_000;
pub const COMPONENT_AUDIT_MAX_FINDINGS: usize = 100_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkComponentKindV1 {
    UserTask,
    Environment,
    GroundTruth,
    Evaluation,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RequirementKindV1 {
    Must,
    MustNot,
    Boundary,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RequirementDispositionV1 {
    Explicit,
    Inferred,
    Ambiguous,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequirementRecordV1 {
    pub requirement_id: String,
    pub source_span_sha256: String,
    pub criterion_sha256: String,
    pub kind: RequirementKindV1,
    pub disposition: RequirementDispositionV1,
    pub environment_precondition_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UserTaskComponentV1 {
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub revision_sha256: String,
    pub instruction_sha256: String,
    pub user_goal_sha256: String,
    pub user_role_sha256: String,
    pub requirements: Vec<RequirementRecordV1>,
    pub required_capability_ids: Vec<String>,
    pub prohibited_capability_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentComponentV1 {
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub revision_sha256: String,
    pub runtime_image_sha256: String,
    pub initial_state_sha256: String,
    pub tool_manifest_sha256: String,
    pub setup_receipt_sha256: String,
    pub teardown_receipt_sha256: String,
    pub provided_capability_ids: Vec<String>,
    pub precondition_ids: Vec<String>,
    pub nondeterminism_policy_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GroundTruthComponentV1 {
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub revision_sha256: String,
    pub oracle_sha256: String,
    pub reference_implementation_sha256: String,
    pub expected_outcomes_sha256: String,
    pub provenance_evidence_sha256: String,
    pub known_limitations_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationComponentV1 {
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub revision_sha256: String,
    pub grader_binary_sha256: String,
    pub grader_source_sha256: String,
    pub rubric_sha256: String,
    pub metric_definition_sha256: String,
    pub thresholds_sha256: String,
    pub blocking_rules_sha256: String,
    pub score_report_schema_sha256: String,
    pub check_ids: Vec<String>,
    pub aggregation_policy_sha256: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ComponentInterfaceKindV1 {
    UserToEnvironment,
    EnvironmentToGroundTruth,
    GroundTruthToEvaluation,
    EvaluationToUser,
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentInterfaceV1 {
    pub interface_id: String,
    pub kind: ComponentInterfaceKindV1,
    pub source_component: BenchmarkComponentKindV1,
    pub target_component: BenchmarkComponentKindV1,
    pub contract_sha256: String,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorCheckV1 {
    pub check_id: String,
    pub check_artifact_sha256: String,
    pub covered_requirement_ids: Vec<String>,
    pub covered_precondition_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComponentDefectClassV1 {
    MissingOrStaleArtifact,
    ArtifactDigestMismatch,
    TaskEnvironmentMismatch,
    CapabilityMismatch,
    AmbiguousRequirement,
    GroundTruthProvenance,
    OracleMismatch,
    GraderBlindSpot,
    ThresholdMismatch,
    MetricAggregationMismatch,
    NondeterministicEnvironment,
    UncoveredRequirement,
    UnjustifiedEvaluatorCheck,
    CrossComponentContradiction,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComponentFindingSeverityV1 {
    Blocker,
    Major,
    Minor,
    Informational,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComponentReviewMethodV1 {
    DeterministicRule,
    HumanReview,
    ModelTriage,
    ExternalAudit,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComponentFindingDispositionV1 {
    Proposed,
    HumanConfirmed,
    Rejected,
    Resolved,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkComponentReviewFindingV1 {
    pub finding_id: String,
    pub component: BenchmarkComponentKindV1,
    pub defect_class: ComponentDefectClassV1,
    pub severity: ComponentFindingSeverityV1,
    pub evidence_sha256: String,
    pub affected_requirement_ids: Vec<String>,
    pub review_method: ComponentReviewMethodV1,
    pub reviewer_identity_commitment_sha256: String,
    /// Digest of the external review/attestation bytes; not authenticated here.
    pub reviewer_attestation_sha256: String,
    pub disposition: ComponentFindingDispositionV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkComponentManifestV1 {
    pub schema: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub task_universe_sha256: String,
    pub user_task: UserTaskComponentV1,
    pub environment: EnvironmentComponentV1,
    pub ground_truth: GroundTruthComponentV1,
    pub evaluation: EvaluationComponentV1,
    pub interfaces: Vec<ComponentInterfaceV1>,
    pub evaluator_checks: Vec<EvaluatorCheckV1>,
    pub review_findings: Vec<BenchmarkComponentReviewFindingV1>,
    pub manifest_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkComponentAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub manifest_digest_matches: bool,
    pub all_four_components_present: bool,
    pub component_artifact_digests_valid: bool,
    pub evaluation_check_bindings_valid: bool,
    pub all_required_interfaces_present: bool,
    pub required_capabilities_available: bool,
    pub prohibited_capabilities_absent: bool,
    pub all_explicit_requirements_covered: bool,
    pub all_requirement_preconditions_covered: bool,
    pub no_orphan_evaluator_checks: bool,
    pub no_orphan_precondition_mappings: bool,
    pub no_unknown_requirement_references: bool,
    pub no_unknown_precondition_references: bool,
    pub review_dispositions_valid: bool,
    pub ambiguous_requirement_count: u64,
    pub inferred_requirement_count: u64,
    pub uncovered_explicit_requirement_count: u64,
    pub uncovered_precondition_count: u64,
    pub unknown_precondition_reference_count: u64,
    pub missing_capability_count: u64,
    pub exposed_prohibited_capability_count: u64,
    pub orphan_evaluator_check_count: u64,
    pub orphan_precondition_mapping_count: u64,
    pub unknown_requirement_reference_count: u64,
    pub review_finding_count: u64,
    pub unresolved_blocker_count: u64,
    pub model_triage_finding_count: u64,
    pub reviewer_attestations_authenticated: bool,
    pub semantic_component_validity_verified: bool,
    pub qualified_for_confirmatory_use: bool,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub manifest_sha256: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
    pub report_sha256: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be lowercase SHA-256 hex"))
    }
}

fn validate_id(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.len() > 256 {
        Err(format!("{label} must be nonempty and at most 256 bytes"))
    } else {
        Ok(())
    }
}

pub fn benchmark_component_manifest_digest(
    manifest: &BenchmarkComponentManifestV1,
) -> Result<String, String> {
    let mut normalized = manifest.clone();
    normalized.manifest_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize component manifest: {error}"))?;
    Ok(hash_domain(b"benchmark-component-manifest.v1", &bytes))
}

fn audit_report_digest(report: &BenchmarkComponentAuditV1) -> Result<String, String> {
    let mut normalized = report.clone();
    normalized.report_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize component audit report: {error}"))?;
    Ok(hash_domain(b"benchmark-component-audit.v1", &bytes))
}

fn validate_component_digests(manifest: &BenchmarkComponentManifestV1) -> Result<(), String> {
    for (digest, label) in [
        (&manifest.task_universe_sha256, "task_universe_sha256"),
        (&manifest.user_task.artifact_sha256, "user_task.artifact_sha256"),
        (&manifest.user_task.revision_sha256, "user_task.revision_sha256"),
        (&manifest.user_task.instruction_sha256, "user_task.instruction_sha256"),
        (&manifest.user_task.user_goal_sha256, "user_task.user_goal_sha256"),
        (&manifest.user_task.user_role_sha256, "user_task.user_role_sha256"),
        (&manifest.environment.artifact_sha256, "environment.artifact_sha256"),
        (&manifest.environment.revision_sha256, "environment.revision_sha256"),
        (&manifest.environment.runtime_image_sha256, "environment.runtime_image_sha256"),
        (&manifest.environment.initial_state_sha256, "environment.initial_state_sha256"),
        (&manifest.environment.tool_manifest_sha256, "environment.tool_manifest_sha256"),
        (&manifest.environment.setup_receipt_sha256, "environment.setup_receipt_sha256"),
        (&manifest.environment.teardown_receipt_sha256, "environment.teardown_receipt_sha256"),
        (&manifest.environment.nondeterminism_policy_sha256, "environment.nondeterminism_policy_sha256"),
        (&manifest.ground_truth.artifact_sha256, "ground_truth.artifact_sha256"),
        (&manifest.ground_truth.revision_sha256, "ground_truth.revision_sha256"),
        (&manifest.ground_truth.oracle_sha256, "ground_truth.oracle_sha256"),
        (&manifest.ground_truth.reference_implementation_sha256, "ground_truth.reference_implementation_sha256"),
        (&manifest.ground_truth.expected_outcomes_sha256, "ground_truth.expected_outcomes_sha256"),
        (&manifest.ground_truth.provenance_evidence_sha256, "ground_truth.provenance_evidence_sha256"),
        (&manifest.ground_truth.known_limitations_sha256, "ground_truth.known_limitations_sha256"),
        (&manifest.evaluation.artifact_sha256, "evaluation.artifact_sha256"),
        (&manifest.evaluation.revision_sha256, "evaluation.revision_sha256"),
        (&manifest.evaluation.grader_binary_sha256, "evaluation.grader_binary_sha256"),
        (&manifest.evaluation.grader_source_sha256, "evaluation.grader_source_sha256"),
        (&manifest.evaluation.rubric_sha256, "evaluation.rubric_sha256"),
        (&manifest.evaluation.metric_definition_sha256, "evaluation.metric_definition_sha256"),
        (&manifest.evaluation.thresholds_sha256, "evaluation.thresholds_sha256"),
        (&manifest.evaluation.blocking_rules_sha256, "evaluation.blocking_rules_sha256"),
        (&manifest.evaluation.score_report_schema_sha256, "evaluation.score_report_schema_sha256"),
        (&manifest.evaluation.aggregation_policy_sha256, "evaluation.aggregation_policy_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    for requirement in &manifest.user_task.requirements {
        validate_digest(&requirement.source_span_sha256, "requirement.source_span_sha256")?;
        validate_digest(&requirement.criterion_sha256, "requirement.criterion_sha256")?;
    }
    for interface in &manifest.interfaces {
        validate_digest(&interface.contract_sha256, "interface.contract_sha256")?;
        validate_digest(&interface.evidence_sha256, "interface.evidence_sha256")?;
    }
    for check in &manifest.evaluator_checks {
        validate_digest(&check.check_artifact_sha256, "check.check_artifact_sha256")?;
    }
    for finding in &manifest.review_findings {
        validate_digest(&finding.evidence_sha256, "finding.evidence_sha256")?;
        validate_digest(&finding.reviewer_identity_commitment_sha256, "finding.reviewer_identity_commitment_sha256")?;
        validate_digest(&finding.reviewer_attestation_sha256, "finding.reviewer_attestation_sha256")?;
    }
    Ok(())
}

fn expected_interfaces() -> BTreeSet<(ComponentInterfaceKindV1, BenchmarkComponentKindV1, BenchmarkComponentKindV1)> {
    [
        (ComponentInterfaceKindV1::UserToEnvironment, BenchmarkComponentKindV1::UserTask, BenchmarkComponentKindV1::Environment),
        (ComponentInterfaceKindV1::EnvironmentToGroundTruth, BenchmarkComponentKindV1::Environment, BenchmarkComponentKindV1::GroundTruth),
        (ComponentInterfaceKindV1::GroundTruthToEvaluation, BenchmarkComponentKindV1::GroundTruth, BenchmarkComponentKindV1::Evaluation),
        (ComponentInterfaceKindV1::EvaluationToUser, BenchmarkComponentKindV1::Evaluation, BenchmarkComponentKindV1::UserTask),
    ].into_iter().collect()
}

pub fn audit_benchmark_components(
    manifest: &BenchmarkComponentManifestV1,
) -> Result<BenchmarkComponentAuditV1, String> {
    if manifest.schema != COMPONENT_MANIFEST_SCHEMA {
        return Err(format!("manifest schema must be {COMPONENT_MANIFEST_SCHEMA}"));
    }
    validate_id(&manifest.benchmark_id, "benchmark_id")?;
    validate_id(&manifest.benchmark_version, "benchmark_version")?;
    validate_id(&manifest.user_task.artifact_id, "user_task.artifact_id")?;
    validate_id(&manifest.environment.artifact_id, "environment.artifact_id")?;
    validate_id(&manifest.ground_truth.artifact_id, "ground_truth.artifact_id")?;
    validate_id(&manifest.evaluation.artifact_id, "evaluation.artifact_id")?;
    validate_digest(&manifest.manifest_sha256, "manifest_sha256")?;
    validate_component_digests(manifest)?;
    if manifest.user_task.requirements.is_empty() || manifest.evaluator_checks.is_empty() {
        return Err("component audit requires at least one user requirement and evaluator check".to_owned());
    }
    if manifest.user_task.requirements.len() > COMPONENT_AUDIT_MAX_REQUIREMENTS
        || manifest.evaluator_checks.len() > COMPONENT_AUDIT_MAX_CHECKS
        || manifest.review_findings.len() > COMPONENT_AUDIT_MAX_FINDINGS
    {
        return Err("component manifest exceeds requirement/check/finding safety bounds".to_owned());
    }
    let manifest_digest_matches =
        benchmark_component_manifest_digest(manifest)? == manifest.manifest_sha256;
    if !manifest_digest_matches {
        return Err("benchmark component manifest digest does not match contents".to_owned());
    }

    let mut issues = Vec::new();
    let mut all_components_present = true;
    let component_artifact_digests_valid = true;
    let mut evaluation_check_bindings_valid = true;
    let mut all_interfaces_present = true;
    let mut required_capabilities_available = true;
    let mut prohibited_capabilities_absent = true;
    let mut all_explicit_requirements_covered = true;
    let mut all_requirement_preconditions_covered = true;
    let mut no_orphan_checks = true;
    let mut no_orphan_precondition_mappings = true;
    let mut no_unknown_requirement_refs = true;
    let mut no_unknown_precondition_refs = true;
    let mut review_dispositions_valid = true;

    let component_ids = [
        manifest.user_task.artifact_id.as_str(),
        manifest.environment.artifact_id.as_str(),
        manifest.ground_truth.artifact_id.as_str(),
        manifest.evaluation.artifact_id.as_str(),
    ];
    let component_id_set: BTreeSet<&str> = component_ids.into_iter().collect();
    if component_id_set.len() != 4 {
        all_components_present = false;
        issues.push("four component artifact IDs must be distinct".to_owned());
    }
    if manifest.interfaces.len() > 10_000 {
        return Err("component interface count exceeds safety bound".to_owned());
    }
    let mut interface_keys = BTreeSet::new();
    let mut interface_ids = BTreeSet::new();
    for interface in &manifest.interfaces {
        validate_id(&interface.interface_id, "interface.interface_id")?;
        if !interface_ids.insert(interface.interface_id.as_str())
            || !interface_keys.insert((interface.kind, interface.source_component, interface.target_component))
        {
            all_interfaces_present = false;
            issues.push(format!("duplicate component interface {}", interface.interface_id));
        }
    }
    let actual_interfaces: BTreeSet<_> = manifest.interfaces.iter()
        .map(|interface| (interface.kind, interface.source_component, interface.target_component))
        .collect();
    if actual_interfaces != expected_interfaces() {
        all_interfaces_present = false;
        issues.push("required four component interfaces are missing or have mismatched endpoints".to_owned());
    }

    let mut requirement_by_id = BTreeMap::new();
    let mut ambiguous_count = 0_u64;
    let mut inferred_count = 0_u64;
    for requirement in &manifest.user_task.requirements {
        validate_id(&requirement.requirement_id, "requirement.requirement_id")?;
        if requirement_by_id.insert(requirement.requirement_id.as_str(), requirement).is_some() {
            return Err("duplicate requirement IDs are not allowed".to_owned());
        }
        let preconditions: BTreeSet<&str> = requirement.environment_precondition_ids.iter()
            .map(String::as_str).collect();
        if preconditions.len() != requirement.environment_precondition_ids.len()
            || preconditions.iter().any(|id| id.trim().is_empty())
        {
            return Err(format!("requirement {} has duplicate or empty precondition IDs", requirement.requirement_id));
        }
        match requirement.disposition {
            RequirementDispositionV1::Ambiguous => ambiguous_count += 1,
            RequirementDispositionV1::Inferred => inferred_count += 1,
            RequirementDispositionV1::Explicit => {}
        }
    }

    let environment_preconditions: BTreeSet<&str> =
        manifest.environment.precondition_ids.iter().map(String::as_str).collect();
    if environment_preconditions.len() != manifest.environment.precondition_ids.len()
        || environment_preconditions.iter().any(|id| id.trim().is_empty())
    {
        return Err("environment precondition IDs must be unique and nonempty".to_owned());
    }
    let provided_capabilities: BTreeSet<&str> =
        manifest.environment.provided_capability_ids.iter().map(String::as_str).collect();
    let required_capabilities: BTreeSet<&str> =
        manifest.user_task.required_capability_ids.iter().map(String::as_str).collect();
    let prohibited_capabilities: BTreeSet<&str> =
        manifest.user_task.prohibited_capability_ids.iter().map(String::as_str).collect();
    if provided_capabilities.len() != manifest.environment.provided_capability_ids.len()
        || required_capabilities.len() != manifest.user_task.required_capability_ids.len()
        || prohibited_capabilities.len() != manifest.user_task.prohibited_capability_ids.len()
        || provided_capabilities.iter().chain(required_capabilities.iter()).chain(prohibited_capabilities.iter())
            .any(|id| id.trim().is_empty())
    {
        return Err("capability IDs must be unique and nonempty within each component".to_owned());
    }
    let missing_capabilities: BTreeSet<_> =
        required_capabilities.difference(&provided_capabilities).copied().collect();
    let exposed_prohibited: BTreeSet<_> =
        prohibited_capabilities.intersection(&provided_capabilities).copied().collect();
    if !missing_capabilities.is_empty() {
        required_capabilities_available = false;
        issues.push(format!("environment lacks {} user-required capabilities", missing_capabilities.len()));
    }
    if !exposed_prohibited.is_empty() {
        prohibited_capabilities_absent = false;
        issues.push(format!("environment exposes {} user-prohibited capabilities", exposed_prohibited.len()));
    }

    let mut covered_requirements: BTreeSet<&str> = BTreeSet::new();
    let mut covered_requirement_preconditions: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut check_ids = BTreeSet::new();
    let mut unknown_requirement_reference_count = 0_u64;
    let mut unknown_precondition_reference_count = 0_u64;
    let mut orphan_check_count = 0_u64;
    let mut orphan_precondition_mapping_count = 0_u64;
    for check in &manifest.evaluator_checks {
        validate_id(&check.check_id, "evaluator_check.check_id")?;
        if !check_ids.insert(check.check_id.as_str()) {
            return Err("duplicate evaluator check IDs are not allowed".to_owned());
        }
        if check.covered_requirement_ids.is_empty() {
            no_orphan_checks = false;
            orphan_check_count += 1;
            issues.push(format!("evaluator check {} has no requirement mapping", check.check_id));
        }
        let mut unique_requirements = BTreeSet::new();
        let mut known_requirements_for_check = Vec::new();
        for requirement_id in &check.covered_requirement_ids {
            if !unique_requirements.insert(requirement_id.as_str()) {
                return Err(format!("evaluator check {} duplicates a requirement mapping", check.check_id));
            }
            if let Some(requirement) = requirement_by_id.get(requirement_id.as_str()) {
                covered_requirements.insert(requirement_id.as_str());
                known_requirements_for_check.push(*requirement);
            } else {
                no_unknown_requirement_refs = false;
                unknown_requirement_reference_count += 1;
                issues.push(format!("evaluator check {} references unknown requirement {}", check.check_id, requirement_id));
            }
        }
        let mut unique_preconditions = BTreeSet::new();
        let mut known_preconditions_for_check = Vec::new();
        for precondition_id in &check.covered_precondition_ids {
            if !unique_preconditions.insert(precondition_id.as_str()) {
                return Err(format!("evaluator check {} duplicates a precondition mapping", check.check_id));
            }
            if environment_preconditions.contains(precondition_id.as_str()) {
                known_preconditions_for_check.push(precondition_id.as_str());
            } else {
                no_unknown_precondition_refs = false;
                unknown_precondition_reference_count += 1;
                issues.push(format!("evaluator check {} references unknown environment precondition {}", check.check_id, precondition_id));
            }
        }
        let mut relevant_preconditions_for_check = BTreeSet::new();
        for requirement in known_requirements_for_check {
            for precondition in &requirement.environment_precondition_ids {
                relevant_preconditions_for_check.insert(precondition.as_str());
                if known_preconditions_for_check.contains(&precondition.as_str()) {
                    covered_requirement_preconditions.insert((
                        requirement.requirement_id.as_str(),
                        precondition.as_str(),
                    ));
                }
            }
        }
        for precondition in &known_preconditions_for_check {
            if !relevant_preconditions_for_check.contains(precondition) {
                no_orphan_precondition_mappings = false;
                orphan_precondition_mapping_count += 1;
                issues.push(format!(
                    "evaluator check {} maps environment precondition {} without a requirement that declares it",
                    check.check_id,
                    precondition
                ));
            }
        }
    }
    let declared_check_ids: BTreeSet<&str> =
        manifest.evaluation.check_ids.iter().map(String::as_str).collect();
    if declared_check_ids.len() != manifest.evaluation.check_ids.len()
        || declared_check_ids.iter().any(|id| id.trim().is_empty())
        || check_ids != declared_check_ids
    {
        evaluation_check_bindings_valid = false;
        issues.push("evaluation component check_ids must be unique, nonempty and exactly match evaluator check records".to_owned());
    }
    let mut uncovered_explicit_count = 0_u64;
    let mut uncovered_precondition_count = 0_u64;
    for requirement in &manifest.user_task.requirements {
        if requirement.disposition == RequirementDispositionV1::Explicit
            && !covered_requirements.contains(requirement.requirement_id.as_str())
        {
            all_explicit_requirements_covered = false;
            uncovered_explicit_count += 1;
            issues.push(format!("explicit requirement {} is not covered by an evaluator check", requirement.requirement_id));
        }
        for precondition in &requirement.environment_precondition_ids {
            if !environment_preconditions.contains(precondition.as_str())
                || !covered_requirement_preconditions.contains(&(
                    requirement.requirement_id.as_str(),
                    precondition.as_str(),
                ))
            {
                all_requirement_preconditions_covered = false;
                uncovered_precondition_count += 1;
                issues.push(format!("requirement {} precondition {} is not declared and checked", requirement.requirement_id, precondition));
            }
        }
    }

    let mut finding_ids = BTreeSet::new();
    let mut unresolved_blockers = 0_u64;
    let mut model_triage = 0_u64;
    for finding in &manifest.review_findings {
        validate_id(&finding.finding_id, "review_finding.finding_id")?;
        if !finding_ids.insert(finding.finding_id.as_str()) {
            return Err("review finding IDs must be unique".to_owned());
        }
        let affected_requirement_ids: BTreeSet<&str> =
            finding.affected_requirement_ids.iter().map(String::as_str).collect();
        if affected_requirement_ids.len() != finding.affected_requirement_ids.len()
            || affected_requirement_ids.iter().any(|id| id.trim().is_empty())
        {
            return Err(format!("review finding {} has duplicate or empty requirement references", finding.finding_id));
        }
        if finding.affected_requirement_ids.iter().any(|id| !requirement_by_id.contains_key(id.as_str())) {
            no_unknown_requirement_refs = false;
            unknown_requirement_reference_count += 1;
            issues.push(format!("review finding {} references an unknown requirement", finding.finding_id));
        }
        if finding.review_method == ComponentReviewMethodV1::ModelTriage {
            model_triage += 1;
            if finding.disposition == ComponentFindingDispositionV1::HumanConfirmed {
                review_dispositions_valid = false;
                issues.push(format!("model-triage finding {} cannot self-assert human confirmation", finding.finding_id));
            }
        }
        if finding.severity == ComponentFindingSeverityV1::Blocker
            && finding.disposition != ComponentFindingDispositionV1::Resolved
            && finding.disposition != ComponentFindingDispositionV1::Rejected
        {
            unresolved_blockers += 1;
        }
    }

    let valid = all_components_present
        && component_artifact_digests_valid
        && evaluation_check_bindings_valid
        && all_interfaces_present
        && required_capabilities_available
        && prohibited_capabilities_absent
        && all_explicit_requirements_covered
        && all_requirement_preconditions_covered
        && no_orphan_checks
        && no_orphan_precondition_mappings
        && no_unknown_requirement_refs
        && no_unknown_precondition_refs
        && review_dispositions_valid
        && unresolved_blockers == 0;
    let mut report = BenchmarkComponentAuditV1 {
        schema: COMPONENT_AUDIT_SCHEMA.to_owned(),
        valid,
        manifest_digest_matches,
        all_four_components_present: all_components_present,
        component_artifact_digests_valid,
        evaluation_check_bindings_valid,
        all_required_interfaces_present: all_interfaces_present,
        required_capabilities_available,
        prohibited_capabilities_absent,
        all_explicit_requirements_covered,
        all_requirement_preconditions_covered,
        no_orphan_evaluator_checks: no_orphan_checks,
        no_orphan_precondition_mappings,
        no_unknown_requirement_references: no_unknown_requirement_refs,
        no_unknown_precondition_references: no_unknown_precondition_refs,
        review_dispositions_valid,
        ambiguous_requirement_count: ambiguous_count,
        inferred_requirement_count: inferred_count,
        uncovered_explicit_requirement_count: uncovered_explicit_count,
        uncovered_precondition_count,
        unknown_precondition_reference_count,
        missing_capability_count: u64::try_from(missing_capabilities.len()).map_err(|_| "missing capability count exceeds u64")?,
        exposed_prohibited_capability_count: u64::try_from(exposed_prohibited.len()).map_err(|_| "prohibited capability count exceeds u64")?,
        orphan_evaluator_check_count: orphan_check_count,
        orphan_precondition_mapping_count,
        unknown_requirement_reference_count,
        review_finding_count: u64::try_from(manifest.review_findings.len()).map_err(|_| "review finding count exceeds u64")?,
        unresolved_blocker_count: unresolved_blockers,
        model_triage_finding_count: model_triage,
        reviewer_attestations_authenticated: false,
        semantic_component_validity_verified: false,
        qualified_for_confirmatory_use: false,
        benchmark_id: manifest.benchmark_id.clone(),
        benchmark_version: manifest.benchmark_version.clone(),
        manifest_sha256: manifest.manifest_sha256.clone(),
        issues,
        claim_boundary: "validates_declared_artifact_bindings_interfaces_and_requirement_coverage_only;semantic_component_validity_and_reviewer_authentication_not_verified".to_owned(),
        report_sha256: String::new(),
    };
    report.report_sha256 = audit_report_digest(&report)?;
    Ok(report)
}

pub fn verify_benchmark_component_audit_integrity(
    report: &BenchmarkComponentAuditV1,
) -> Result<(), String> {
    if report.schema != COMPONENT_AUDIT_SCHEMA
        || !report.manifest_digest_matches
        || report.reviewer_attestations_authenticated
        || report.semantic_component_validity_verified
        || report.qualified_for_confirmatory_use
    {
        return Err("component audit schema or trust boundary is invalid".to_owned());
    }
    validate_digest(&report.manifest_sha256, "manifest_sha256")?;
    validate_digest(&report.report_sha256, "report_sha256")?;
    if audit_report_digest(report)? != report.report_sha256 {
        return Err("benchmark component audit report digest mismatch".to_owned());
    }
    let summary_consistent =
        report.all_explicit_requirements_covered
            == (report.uncovered_explicit_requirement_count == 0)
        && report.all_requirement_preconditions_covered
            == (report.uncovered_precondition_count == 0)
        && report.required_capabilities_available
            == (report.missing_capability_count == 0)
        && report.prohibited_capabilities_absent
            == (report.exposed_prohibited_capability_count == 0)
        && report.no_orphan_evaluator_checks
            == (report.orphan_evaluator_check_count == 0)
        && report.no_orphan_precondition_mappings
            == (report.orphan_precondition_mapping_count == 0)
        && report.no_unknown_requirement_references
            == (report.unknown_requirement_reference_count == 0)
        && report.no_unknown_precondition_references
            == (report.unknown_precondition_reference_count == 0)
        && (report.unresolved_blocker_count == 0
            || !report.valid);
    if !summary_consistent {
        return Err("benchmark component audit summary counts disagree with status flags".to_owned());
    }
    if report.valid
        != (report.manifest_digest_matches
            && report.all_four_components_present
            && report.component_artifact_digests_valid
            && report.evaluation_check_bindings_valid
            && report.all_required_interfaces_present
            && report.required_capabilities_available
            && report.prohibited_capabilities_absent
            && report.all_explicit_requirements_covered
            && report.all_requirement_preconditions_covered
            && report.no_orphan_evaluator_checks
            && report.no_orphan_precondition_mappings
            && report.no_unknown_requirement_references
            && report.no_unknown_precondition_references
            && report.review_dispositions_valid
            && report.unresolved_blocker_count == 0)
    {
        return Err("benchmark component audit report validity does not match its checks".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn digest(label: &str) -> String {
        hash_domain(b"benchmark-component-test", label.as_bytes())
    }

    fn manifest() -> BenchmarkComponentManifestV1 {
        let mut manifest = BenchmarkComponentManifestV1 {
            schema: COMPONENT_MANIFEST_SCHEMA.to_owned(),
            benchmark_id: "synthetic-agent-bench".to_owned(),
            benchmark_version: "v1".to_owned(),
            task_universe_sha256: digest("task-universe"),
            user_task: UserTaskComponentV1 {
                artifact_id: "user-task-v1".to_owned(),
                artifact_sha256: digest("user-artifact"),
                revision_sha256: digest("user-revision"),
                instruction_sha256: digest("instruction"),
                user_goal_sha256: digest("goal"),
                user_role_sha256: digest("role"),
                requirements: vec![
                    RequirementRecordV1 {
                        requirement_id: "REQ-001".to_owned(),
                        source_span_sha256: digest("span-1"),
                        criterion_sha256: digest("criterion-1"),
                        kind: RequirementKindV1::Must,
                        disposition: RequirementDispositionV1::Explicit,
                        environment_precondition_ids: vec!["PRE-001".to_owned()],
                    },
                    RequirementRecordV1 {
                        requirement_id: "REQ-002".to_owned(),
                        source_span_sha256: digest("span-2"),
                        criterion_sha256: digest("criterion-2"),
                        kind: RequirementKindV1::MustNot,
                        disposition: RequirementDispositionV1::Explicit,
                        environment_precondition_ids: vec![],
                    },
                    RequirementRecordV1 {
                        requirement_id: "REQ-003".to_owned(),
                        source_span_sha256: digest("span-3"),
                        criterion_sha256: digest("criterion-3"),
                        kind: RequirementKindV1::Boundary,
                        disposition: RequirementDispositionV1::Ambiguous,
                        environment_precondition_ids: vec![],
                    },
                ],
                required_capability_ids: vec!["read_state".to_owned()],
                prohibited_capability_ids: vec!["write_external".to_owned()],
            },
            environment: EnvironmentComponentV1 {
                artifact_id: "environment-v1".to_owned(),
                artifact_sha256: digest("environment-artifact"),
                revision_sha256: digest("environment-revision"),
                runtime_image_sha256: digest("runtime"),
                initial_state_sha256: digest("initial-state"),
                tool_manifest_sha256: digest("tool-manifest"),
                setup_receipt_sha256: digest("setup"),
                teardown_receipt_sha256: digest("teardown"),
                provided_capability_ids: vec!["read_state".to_owned()],
                precondition_ids: vec!["PRE-001".to_owned()],
                nondeterminism_policy_sha256: digest("nondeterminism-policy"),
            },
            ground_truth: GroundTruthComponentV1 {
                artifact_id: "ground-truth-v1".to_owned(),
                artifact_sha256: digest("ground-truth-artifact"),
                revision_sha256: digest("ground-truth-revision"),
                oracle_sha256: digest("oracle"),
                reference_implementation_sha256: digest("reference"),
                expected_outcomes_sha256: digest("outcomes"),
                provenance_evidence_sha256: digest("provenance"),
                known_limitations_sha256: digest("limitations"),
            },
            evaluation: EvaluationComponentV1 {
                artifact_id: "evaluation-v1".to_owned(),
                artifact_sha256: digest("evaluation-artifact"),
                revision_sha256: digest("evaluation-revision"),
                grader_binary_sha256: digest("grader-binary"),
                grader_source_sha256: digest("grader-source"),
                rubric_sha256: digest("rubric"),
                metric_definition_sha256: digest("metric"),
                thresholds_sha256: digest("thresholds"),
                blocking_rules_sha256: digest("blocking-rules"),
                score_report_schema_sha256: digest("score-schema"),
                check_ids: vec!["CHECK-001".to_owned(), "CHECK-002".to_owned()],
                aggregation_policy_sha256: digest("aggregation"),
            },
            interfaces: vec![
                ComponentInterfaceV1 {
                    interface_id: "IF-USER-ENV".to_owned(),
                    kind: ComponentInterfaceKindV1::UserToEnvironment,
                    source_component: BenchmarkComponentKindV1::UserTask,
                    target_component: BenchmarkComponentKindV1::Environment,
                    contract_sha256: digest("user-env-contract"),
                    evidence_sha256: digest("user-env-evidence"),
                },
                ComponentInterfaceV1 {
                    interface_id: "IF-ENV-GT".to_owned(),
                    kind: ComponentInterfaceKindV1::EnvironmentToGroundTruth,
                    source_component: BenchmarkComponentKindV1::Environment,
                    target_component: BenchmarkComponentKindV1::GroundTruth,
                    contract_sha256: digest("env-gt-contract"),
                    evidence_sha256: digest("env-gt-evidence"),
                },
                ComponentInterfaceV1 {
                    interface_id: "IF-GT-EVAL".to_owned(),
                    kind: ComponentInterfaceKindV1::GroundTruthToEvaluation,
                    source_component: BenchmarkComponentKindV1::GroundTruth,
                    target_component: BenchmarkComponentKindV1::Evaluation,
                    contract_sha256: digest("gt-eval-contract"),
                    evidence_sha256: digest("gt-eval-evidence"),
                },
                ComponentInterfaceV1 {
                    interface_id: "IF-EVAL-USER".to_owned(),
                    kind: ComponentInterfaceKindV1::EvaluationToUser,
                    source_component: BenchmarkComponentKindV1::Evaluation,
                    target_component: BenchmarkComponentKindV1::UserTask,
                    contract_sha256: digest("eval-user-contract"),
                    evidence_sha256: digest("eval-user-evidence"),
                },
            ],
            evaluator_checks: vec![
                EvaluatorCheckV1 {
                    check_id: "CHECK-001".to_owned(),
                    check_artifact_sha256: digest("check-1"),
                    covered_requirement_ids: vec!["REQ-001".to_owned()],
                    covered_precondition_ids: vec!["PRE-001".to_owned()],
                },
                EvaluatorCheckV1 {
                    check_id: "CHECK-002".to_owned(),
                    check_artifact_sha256: digest("check-2"),
                    covered_requirement_ids: vec!["REQ-002".to_owned(), "REQ-003".to_owned()],
                    covered_precondition_ids: vec![],
                },
            ],
            review_findings: vec![],
            manifest_sha256: String::new(),
            claim_boundary: "synthetic contract fixture".to_owned(),
        };
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        manifest
    }

    #[test]
    fn strict_component_manifest_and_audit_schemas_match_structs() {
        let manifest_schema: Value = serde_json::from_str(include_str!(
            "../schema/benchmark-component-manifest.schema.json"
        )).unwrap();
        let audit_schema: Value = serde_json::from_str(include_str!(
            "../schema/benchmark-component-audit.schema.json"
        )).unwrap();
        let manifest = manifest();
        let audit = audit_benchmark_components(&manifest).unwrap();
        for (schema, value, version) in [
            (&manifest_schema, serde_json::to_value(&manifest).unwrap(), COMPONENT_MANIFEST_SCHEMA),
            (&audit_schema, serde_json::to_value(&audit).unwrap(), COMPONENT_AUDIT_SCHEMA),
        ] {
            assert_eq!(schema["properties"]["schema"]["const"].as_str(), Some(version));
            let required: BTreeSet<String> = schema["required"].as_array().unwrap()
                .iter().map(|item| item.as_str().unwrap().to_owned()).collect();
            let properties: BTreeSet<String> = schema["properties"].as_object().unwrap()
                .keys().cloned().collect();
            let actual: BTreeSet<String> = value.as_object().unwrap().keys().cloned().collect();
            assert_eq!(required, properties);
            assert_eq!(actual, properties);
        }
        assert!(audit.valid);
        assert!(!audit.qualified_for_confirmatory_use);

        let user_properties: BTreeSet<String> =
            manifest_schema["properties"]["user_task"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_user: BTreeSet<String> = serde_json::to_value(&manifest.user_task)
            .unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(user_properties, actual_user);
        let requirement_properties: BTreeSet<String> =
            manifest_schema["properties"]["user_task"]["properties"]["requirements"]["items"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_requirement: BTreeSet<String> = serde_json::to_value(
            &manifest.user_task.requirements[0]
        ).unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(requirement_properties, actual_requirement);
        let interface_properties: BTreeSet<String> =
            manifest_schema["properties"]["interfaces"]["items"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_interface: BTreeSet<String> = serde_json::to_value(
            &manifest.interfaces[0]
        ).unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(interface_properties, actual_interface);
        let check_properties: BTreeSet<String> =
            manifest_schema["properties"]["evaluator_checks"]["items"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_check: BTreeSet<String> = serde_json::to_value(
            &manifest.evaluator_checks[0]
        ).unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(check_properties, actual_check);

        let env_properties: BTreeSet<String> =
            manifest_schema["properties"]["environment"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_env: BTreeSet<String> = serde_json::to_value(&manifest.environment)
            .unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(env_properties, actual_env);
        let truth_properties: BTreeSet<String> =
            manifest_schema["properties"]["ground_truth"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_truth: BTreeSet<String> = serde_json::to_value(&manifest.ground_truth)
            .unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(truth_properties, actual_truth);
        let eval_properties: BTreeSet<String> =
            manifest_schema["properties"]["evaluation"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_eval: BTreeSet<String> = serde_json::to_value(&manifest.evaluation)
            .unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(eval_properties, actual_eval);

        let mut reviewed_manifest = manifest.clone();
        reviewed_manifest.review_findings.push(BenchmarkComponentReviewFindingV1 {
            finding_id: "FIND-SCHEMA".to_owned(),
            component: BenchmarkComponentKindV1::Evaluation,
            defect_class: ComponentDefectClassV1::GraderBlindSpot,
            severity: ComponentFindingSeverityV1::Minor,
            evidence_sha256: digest("schema-evidence"),
            affected_requirement_ids: vec!["REQ-001".to_owned()],
            review_method: ComponentReviewMethodV1::HumanReview,
            reviewer_identity_commitment_sha256: digest("reviewer"),
            reviewer_attestation_sha256: digest("review-attestation"),
            disposition: ComponentFindingDispositionV1::Proposed,
        });
        reviewed_manifest.manifest_sha256 =
            benchmark_component_manifest_digest(&reviewed_manifest).unwrap();
        let finding_properties: BTreeSet<String> =
            manifest_schema["properties"]["review_findings"]["items"]["properties"]
                .as_object().unwrap().keys().cloned().collect();
        let actual_finding: BTreeSet<String> = serde_json::to_value(
            &reviewed_manifest.review_findings[0]
        ).unwrap().as_object().unwrap().keys().cloned().collect();
        assert_eq!(finding_properties, actual_finding);
    }

    #[test]
    fn rehashed_report_cannot_claim_coverage_with_nonzero_uncovered_count() {
        let manifest = manifest();
        let mut report = audit_benchmark_components(&manifest).unwrap();
        report.uncovered_explicit_requirement_count = 1;
        report.report_sha256.clear();
        report.report_sha256 = audit_report_digest(&report).unwrap();
        assert!(verify_benchmark_component_audit_integrity(&report).unwrap_err()
            .contains("summary counts disagree"));
    }

    #[test]
    fn explicit_uncovered_requirement_is_counted_not_silently_dropped() {
        let mut manifest = manifest();
        manifest.evaluator_checks[0].covered_requirement_ids.clear();
        manifest.evaluation.check_ids = vec!["CHECK-002".to_owned()];
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.valid);
        assert_eq!(report.uncovered_explicit_requirement_count, 1);
        assert_eq!(report.orphan_evaluator_check_count, 1);
    }

    #[test]
    fn required_capability_missing_and_prohibited_capability_present_fail() {
        let mut manifest = manifest();
        manifest.environment.provided_capability_ids = vec![
            "write_external".to_owned(),
        ];
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.required_capabilities_available);
        assert!(!report.prohibited_capabilities_absent);
        assert_eq!(report.missing_capability_count, 1);
        assert_eq!(report.exposed_prohibited_capability_count, 1);
    }

    #[test]
    fn requirement_precondition_must_be_covered_by_same_check() {
        let mut manifest = manifest();
        manifest.evaluator_checks[0].covered_precondition_ids.clear();
        manifest.evaluator_checks[1].covered_precondition_ids = vec!["PRE-001".to_owned()];
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.all_requirement_preconditions_covered);
        assert_eq!(report.uncovered_precondition_count, 1);
    }

    #[test]
    fn missing_or_mismatched_component_interfaces_fail() {
        let mut manifest = manifest();
        manifest.interfaces.pop();
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.all_required_interfaces_present);
        assert!(!report.valid);

        let mut manifest = manifest();
        manifest.interfaces[0].target_component = BenchmarkComponentKindV1::GroundTruth;
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        assert!(!audit_benchmark_components(&manifest).unwrap().all_required_interfaces_present);
    }

    #[test]
    fn unknown_requirement_and_precondition_references_fail_closed() {
        let mut manifest = manifest();
        manifest.evaluator_checks[0].covered_requirement_ids.push("REQ-UNKNOWN".to_owned());
        manifest.evaluator_checks[0].covered_precondition_ids.push("PRE-UNKNOWN".to_owned());
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.no_unknown_requirement_references);
        assert!(!report.no_unknown_precondition_references);
        assert!(report.all_requirement_preconditions_covered);
        assert_eq!(report.unknown_requirement_reference_count, 1);
        assert_eq!(report.unknown_precondition_reference_count, 1);
    }

    #[test]
    fn stale_manifest_digest_is_rejected_before_audit() {
        let mut manifest = manifest();
        manifest.user_task.instruction_sha256 = digest("changed-instruction");
        assert!(audit_benchmark_components(&manifest).unwrap_err()
            .contains("manifest digest does not match"));
    }

    #[test]
    fn duplicate_requirement_check_and_interface_ids_are_rejected() {
        let mut manifest = manifest();
        manifest.user_task.requirements.push(manifest.user_task.requirements[0].clone());
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        assert!(audit_benchmark_components(&manifest).unwrap_err().contains("duplicate requirement"));

        let mut manifest = manifest();
        manifest.evaluator_checks[1].check_id = manifest.evaluator_checks[0].check_id.clone();
        manifest.evaluation.check_ids = vec!["CHECK-001".to_owned(), "CHECK-001".to_owned()];
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        assert!(audit_benchmark_components(&manifest).unwrap_err().contains("duplicate evaluator check"));

        let mut manifest = manifest();
        manifest.interfaces[1].interface_id = manifest.interfaces[0].interface_id.clone();
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.all_required_interfaces_present);
    }

    #[test]
    fn evaluator_precondition_mapping_must_belong_to_a_mapped_requirement() {
        let mut manifest = manifest();
        manifest.evaluator_checks[1].covered_precondition_ids = vec!["PRE-001".to_owned()];
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.no_orphan_precondition_mappings);
        assert_eq!(report.orphan_precondition_mapping_count, 1);
        assert!(!report.valid);
    }

    #[test]
    fn empty_requirement_or_check_universe_fails_closed() {
        let mut manifest = manifest();
        manifest.user_task.requirements.clear();
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        assert!(audit_benchmark_components(&manifest).unwrap_err()
            .contains("at least one user requirement"));

        let mut manifest = manifest();
        manifest.evaluator_checks.clear();
        manifest.evaluation.check_ids.clear();
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        assert!(audit_benchmark_components(&manifest).unwrap_err()
            .contains("at least one user requirement"));
    }

    #[test]
    fn model_triage_cannot_self_assert_human_confirmation() {
        let mut manifest = manifest();
        manifest.review_findings.push(BenchmarkComponentReviewFindingV1 {
            finding_id: "FIND-001".to_owned(),
            component: BenchmarkComponentKindV1::Evaluation,
            defect_class: ComponentDefectClassV1::GraderBlindSpot,
            severity: ComponentFindingSeverityV1::Major,
            evidence_sha256: digest("finding-evidence"),
            affected_requirement_ids: vec!["REQ-001".to_owned()],
            review_method: ComponentReviewMethodV1::ModelTriage,
            reviewer_identity_commitment_sha256: digest("model-reviewer"),
            reviewer_attestation_sha256: digest("model-attestation"),
            disposition: ComponentFindingDispositionV1::HumanConfirmed,
        });
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert_eq!(report.model_triage_finding_count, 1);
        assert!(report.issues.iter().any(|issue| issue.contains("cannot self-assert human confirmation")));
        assert!(!report.valid);
        assert!(!report.review_dispositions_valid);
        assert!(!report.reviewer_attestations_authenticated);
        assert!(!report.qualified_for_confirmatory_use);
    }

    #[test]
    fn unresolved_blocker_prevents_valid_audit() {
        let mut manifest = manifest();
        manifest.review_findings.push(BenchmarkComponentReviewFindingV1 {
            finding_id: "FIND-BLOCKER".to_owned(),
            component: BenchmarkComponentKindV1::GroundTruth,
            defect_class: ComponentDefectClassV1::OracleMismatch,
            severity: ComponentFindingSeverityV1::Blocker,
            evidence_sha256: digest("blocker-evidence"),
            affected_requirement_ids: vec!["REQ-001".to_owned()],
            review_method: ComponentReviewMethodV1::HumanReview,
            reviewer_identity_commitment_sha256: digest("reviewer"),
            reviewer_attestation_sha256: digest("review-attestation"),
            disposition: ComponentFindingDispositionV1::HumanConfirmed,
        });
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.valid);
        assert_eq!(report.unresolved_blocker_count, 1);
    }

    #[test]
    fn orphan_check_and_unknown_review_requirement_are_reported() {
        let mut manifest = manifest();
        manifest.evaluator_checks.push(EvaluatorCheckV1 {
            check_id: "CHECK-ORPHAN".to_owned(),
            check_artifact_sha256: digest("orphan-check"),
            covered_requirement_ids: vec![],
            covered_precondition_ids: vec![],
        });
        manifest.evaluation.check_ids.push("CHECK-ORPHAN".to_owned());
        manifest.review_findings.push(BenchmarkComponentReviewFindingV1 {
            finding_id: "FIND-UNKNOWN".to_owned(),
            component: BenchmarkComponentKindV1::UserTask,
            defect_class: ComponentDefectClassV1::Unknown,
            severity: ComponentFindingSeverityV1::Minor,
            evidence_sha256: digest("unknown-evidence"),
            affected_requirement_ids: vec!["REQ-MISSING".to_owned()],
            review_method: ComponentReviewMethodV1::HumanReview,
            reviewer_identity_commitment_sha256: digest("human-reviewer"),
            reviewer_attestation_sha256: digest("human-attestation"),
            disposition: ComponentFindingDispositionV1::Proposed,
        });
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.no_orphan_evaluator_checks);
        assert!(!report.no_unknown_requirement_references);
        assert_eq!(report.orphan_evaluator_check_count, 1);
        assert_eq!(report.unknown_requirement_reference_count, 1);
    }

    #[test]
    fn duplicate_declared_check_ids_fail_component_binding() {
        let mut manifest = manifest();
        manifest.evaluation.check_ids = vec!["CHECK-001".to_owned(), "CHECK-001".to_owned()];
        manifest.manifest_sha256 = benchmark_component_manifest_digest(&manifest).unwrap();
        let report = audit_benchmark_components(&manifest).unwrap();
        assert!(!report.evaluation_check_bindings_valid);
        assert!(!report.valid);
    }

    #[test]
    fn report_integrity_detects_rehashed_validity_flag_tampering() {
        let manifest = manifest();
        let mut report = audit_benchmark_components(&manifest).unwrap();
        report.valid = false;
        report.report_sha256.clear();
        report.report_sha256 = audit_report_digest(&report).unwrap();
        assert!(verify_benchmark_component_audit_integrity(&report).unwrap_err()
            .contains("validity does not match"));
    }
}
