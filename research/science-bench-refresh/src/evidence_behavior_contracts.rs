//! Precommitted evidence-behavior contracts and terminal adjudication receipts.
//!
//! This module validates the declared evaluator contract and outcome accounting.
//! It does not prove that a runner applied an intervention, that an adjudicator
//! is independent, or that an adjudicated label is true. Qualified runs require
//! an independently anchored pre-run commitment and externally auditable receipts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::BUILT_GENERATOR_SOURCE_REVISION;

pub const EVIDENCE_BEHAVIOR_CONTRACT_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_evidence_behavior_contract_plan.v1";
pub const EVIDENCE_BEHAVIOR_CONTRACT_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_evidence_behavior_contract_report.v1";

const HASH_PREFIX: &[u8] =
    b"symthaea.science_bench_refresh.evidence_behavior_contracts.v1\0";
const PLAN_DOMAIN: &[u8] = b"plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"report.v1\0";
const MAX_CASES: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl ContractPurposeV1 {
    fn is_qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractFamilyV1 {
    UpdateCompliance,
    ConflictDisclosure,
    DecoyInvariance,
    DeletionCalibration,
    PermutationInvariance,
    EvidenceForceMonotonicity,
    ClaimEvidenceCompleteness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractInterventionV1 {
    CoherentUpdate,
    SourceConflict,
    IrrelevantDecoy,
    KeyEvidenceDeletion,
    EvidencePermutation,
    EvidenceStrengthReduction,
    JointClaimEvidenceAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedBehaviorV1 {
    DependentClaimsUpdateWithoutUnrelatedDrift,
    ExplicitConflictAndProvenance,
    SupportedClaimsStable,
    ClaimStrengthTracksRemainingSupport,
    SemanticAnswerInvariant,
    WeakerEvidenceNotStrongerClaim,
    AllMaterialClaimsWarranted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForceAxisV1 {
    NotApplicable,
    Relation,
    Modality,
    Scope,
    TemporalValidity,
    NumericSpecificity,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBehaviorContractItemV1 {
    pub case_id: String,
    pub pair_id: String,
    pub family: ContractFamilyV1,
    pub intervention: ContractInterventionV1,
    pub baseline_condition: String,
    pub intervention_condition: String,
    pub expected_behavior: ExpectedBehaviorV1,
    pub force_axis: ForceAxisV1,
    pub decision_rule_id: String,
    pub evidence_manifest_sha256: String,
    pub evaluator_gold_sha256: String,
}

/// Typed plan payload excluding only plan_commitment_sha256. Keep field order
/// stable: the commitment hashes this struct's compact serde_json serialization.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBehaviorContractPlanFieldsV1 {
    pub schema: String,
    pub purpose: ContractPurposeV1,
    pub contract_set_id: String,
    pub evaluator_source_revision: String,
    pub source_authority_policy_sha256: String,
    pub decision_rules_sha256: String,
    pub items: Vec<EvidenceBehaviorContractItemV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBehaviorContractPlanV1 {
    pub schema: String,
    pub purpose: ContractPurposeV1,
    pub contract_set_id: String,
    pub evaluator_source_revision: String,
    pub source_authority_policy_sha256: String,
    pub decision_rules_sha256: String,
    pub items: Vec<EvidenceBehaviorContractItemV1>,
    pub plan_commitment_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractOutcomeStatusV1 {
    Pass,
    Violation,
    NotEvaluable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractAdjudicationModeV1 {
    DeterministicSyntheticOracle,
    BlindedHuman,
    DualHumanResolved,
    AutomatedJudgeBlindedAudit,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBehaviorContractOutcomeRecordV1 {
    pub case_id: String,
    pub status: ContractOutcomeStatusV1,
    pub baseline_response_artifact_sha256: Option<String>,
    pub intervention_response_artifact_sha256: Option<String>,
    pub adjudication_record_sha256: String,
    pub violation_codes: Vec<String>,
    pub not_evaluable_reason: Option<String>,
}

/// Typed report payload excluding only report_commitment_sha256.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBehaviorContractReportFieldsV1 {
    pub schema: String,
    pub purpose: ContractPurposeV1,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub evaluator_source_revision: String,
    pub decision_rules_sha256: String,
    pub adjudication_mode: ContractAdjudicationModeV1,
    pub blind_to_system_identity: bool,
    pub records: Vec<EvidenceBehaviorContractOutcomeRecordV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBehaviorContractReportV1 {
    pub schema: String,
    pub purpose: ContractPurposeV1,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub evaluator_source_revision: String,
    pub decision_rules_sha256: String,
    pub adjudication_mode: ContractAdjudicationModeV1,
    pub blind_to_system_identity: bool,
    pub records: Vec<EvidenceBehaviorContractOutcomeRecordV1>,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContractFamilySummaryV1 {
    pub family: ContractFamilyV1,
    pub planned_cases: u64,
    pub pass_count: u64,
    pub violation_count: u64,
    pub not_evaluable_count: u64,
    pub eligible_count: u64,
    /// Rounded rate in millionths; None means that no cases were eligible.
    pub violation_rate_millionths: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBehaviorContractScorecardV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub report_sha256: String,
    pub report_commitment_sha256: String,
    pub planned_case_count: u64,
    pub observed_record_count: u64,
    pub missing_case_count: u64,
    pub pass_count: u64,
    pub violation_count: u64,
    pub not_evaluable_count: u64,
    pub family_summaries: Vec<ContractFamilySummaryV1>,
    pub fixed_benchmark_only: bool,
    pub generalized_accuracy_estimated: bool,
    pub runner_execution_proven: bool,
    pub external_anchor_required: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBehaviorContractPlanVerificationV1 {
    pub schema: String,
    pub plan_sha256: String,
    pub plan_commitment_expected: String,
    pub plan_commitment_recomputed: String,
    pub plan_commitment_matches: bool,
    pub structural_valid: bool,
    pub valid: bool,
    pub contract_set_id: String,
    pub planned_case_count: u64,
    pub issues: Vec<String>,
    pub scoring_eligible_declared: bool,
    pub external_anchor_required: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBehaviorContractReportVerificationV1 {
    pub schema: String,
    pub valid: bool,
    pub plan_binding_valid: bool,
    pub report_commitment_matches: bool,
    pub planned_case_count: u64,
    pub observed_record_count: u64,
    pub missing_case_count: u64,
    pub issues: Vec<String>,
    pub scorecard: Option<EvidenceBehaviorContractScorecardV1>,
}

#[derive(Debug, Clone, Serialize)]
struct OutcomeCommitmentPayloadV1<'a> {
    schema: &'a str,
    purpose: ContractPurposeV1,
    plan_sha256: &'a str,
    plan_commitment_sha256: &'a str,
    evaluator_source_revision: &'a str,
    decision_rules_sha256: &'a str,
    adjudication_mode: ContractAdjudicationModeV1,
    blind_to_system_identity: bool,
    records: &'a [EvidenceBehaviorContractOutcomeRecordV1],
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value != "0".repeat(64))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value != "0".repeat(40)
}

fn digest_domain_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len =
        u64::try_from(domain.len()).map_err(|_| "hash domain exceeds u64 length".to_owned())?;
    let payload_len =
        u64::try_from(payload.len()).map_err(|_| "hash payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update(domain_len.to_be_bytes());
    hasher.update(domain);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(hex_digest(&hasher.finalize()))
}

fn expected_pair(
    family: ContractFamilyV1,
) -> (ContractInterventionV1, ExpectedBehaviorV1) {
    match family {
        ContractFamilyV1::UpdateCompliance => (
            ContractInterventionV1::CoherentUpdate,
            ExpectedBehaviorV1::DependentClaimsUpdateWithoutUnrelatedDrift,
        ),
        ContractFamilyV1::ConflictDisclosure => (
            ContractInterventionV1::SourceConflict,
            ExpectedBehaviorV1::ExplicitConflictAndProvenance,
        ),
        ContractFamilyV1::DecoyInvariance => (
            ContractInterventionV1::IrrelevantDecoy,
            ExpectedBehaviorV1::SupportedClaimsStable,
        ),
        ContractFamilyV1::DeletionCalibration => (
            ContractInterventionV1::KeyEvidenceDeletion,
            ExpectedBehaviorV1::ClaimStrengthTracksRemainingSupport,
        ),
        ContractFamilyV1::PermutationInvariance => (
            ContractInterventionV1::EvidencePermutation,
            ExpectedBehaviorV1::SemanticAnswerInvariant,
        ),
        ContractFamilyV1::EvidenceForceMonotonicity => (
            ContractInterventionV1::EvidenceStrengthReduction,
            ExpectedBehaviorV1::WeakerEvidenceNotStrongerClaim,
        ),
        ContractFamilyV1::ClaimEvidenceCompleteness => (
            ContractInterventionV1::JointClaimEvidenceAudit,
            ExpectedBehaviorV1::AllMaterialClaimsWarranted,
        ),
    }
}

fn validate_plan_fields(fields: &EvidenceBehaviorContractPlanFieldsV1) -> Vec<String> {
    let mut issues = Vec::new();
    if fields.schema != EVIDENCE_BEHAVIOR_CONTRACT_PLAN_SCHEMA {
        issues.push("unsupported evidence behavior contract plan schema".to_owned());
    }
    if fields.contract_set_id.trim().is_empty() || fields.contract_set_id.len() > 128 {
        issues.push("contract_set_id must be non-empty and at most 128 bytes".to_owned());
    }
    if fields.items.is_empty() || fields.items.len() > MAX_CASES {
        issues.push(format!("items must contain 1..={MAX_CASES} cases"));
    }
    let allow_zero = !fields.purpose.is_qualified();
    if !valid_digest(&fields.source_authority_policy_sha256, allow_zero) {
        issues.push("source_authority_policy_sha256 is not an eligible SHA-256 digest".to_owned());
    }
    if !valid_digest(&fields.decision_rules_sha256, allow_zero) {
        issues.push("decision_rules_sha256 is not an eligible SHA-256 digest".to_owned());
    }
    if fields.purpose.is_qualified() {
        if !valid_revision(&fields.evaluator_source_revision) {
            issues.push("qualified run requires a non-zero lower-case source revision SHA".to_owned());
        } else if fields.evaluator_source_revision != BUILT_GENERATOR_SOURCE_REVISION {
            issues.push("evaluator source revision does not match the inspected build revision".to_owned());
        }
    } else if fields.evaluator_source_revision.trim().is_empty() {
        issues.push("example-only source revision marker must be non-empty".to_owned());
    }

    let mut seen = BTreeSet::new();
    for item in &fields.items {
        if item.case_id.trim().is_empty() || item.case_id.len() > 128 {
            issues.push("case_id must be non-empty and at most 128 bytes".to_owned());
        }
        if !seen.insert(item.case_id.clone()) {
            issues.push(format!("duplicate case_id {}", item.case_id));
        }
        if item.pair_id.trim().is_empty() || item.pair_id.len() > 128 {
            issues.push(format!("case {} has invalid pair_id", item.case_id));
        }
        if item.baseline_condition.trim().is_empty()
            || item.intervention_condition.trim().is_empty()
            || item.baseline_condition == item.intervention_condition
        {
            issues.push(format!("case {} has invalid condition pair", item.case_id));
        }
        if item.decision_rule_id.trim().is_empty() || item.decision_rule_id.len() > 128 {
            issues.push(format!("case {} has invalid decision_rule_id", item.case_id));
        }
        let (expected_intervention, expected_behavior) = expected_pair(item.family);
        if item.intervention != expected_intervention || item.expected_behavior != expected_behavior {
            issues.push(format!(
                "case {} has intervention/behavior incompatible with family {:?}",
                item.case_id, item.family
            ));
        }
        if item.family == ContractFamilyV1::EvidenceForceMonotonicity {
            if item.force_axis == ForceAxisV1::NotApplicable {
                issues.push(format!("force case {} must declare a force axis", item.case_id));
            }
        } else if item.force_axis != ForceAxisV1::NotApplicable {
            issues.push(format!(
                "non-force case {} must use force_axis=not_applicable",
                item.case_id
            ));
        }
        if !valid_digest(&item.evidence_manifest_sha256, allow_zero) {
            issues.push(format!(
                "case {} evidence_manifest_sha256 is not an eligible digest",
                item.case_id
            ));
        }
        if !valid_digest(&item.evaluator_gold_sha256, allow_zero) {
            issues.push(format!(
                "case {} evaluator_gold_sha256 is not an eligible digest",
                item.case_id
            ));
        }
    }
    issues
}

fn fields_from_plan(plan: &EvidenceBehaviorContractPlanV1) -> EvidenceBehaviorContractPlanFieldsV1 {
    EvidenceBehaviorContractPlanFieldsV1 {
        schema: plan.schema.clone(),
        purpose: plan.purpose,
        contract_set_id: plan.contract_set_id.clone(),
        evaluator_source_revision: plan.evaluator_source_revision.clone(),
        source_authority_policy_sha256: plan.source_authority_policy_sha256.clone(),
        decision_rules_sha256: plan.decision_rules_sha256.clone(),
        items: plan.items.clone(),
    }
}

fn require_object_keys(
    value: &serde_json::Value,
    keys: &[&str],
    label: &str,
) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    for key in keys {
        if !object.contains_key(*key) {
            return Err(format!("{label} is missing required field {key}"));
        }
    }
    Ok(())
}

fn validate_plan_json_shape(bytes: &[u8], full_plan: bool) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("contract plan JSON is malformed: {error}"))?;
    let mut keys = vec![
        "schema",
        "purpose",
        "contract_set_id",
        "evaluator_source_revision",
        "source_authority_policy_sha256",
        "decision_rules_sha256",
        "items",
    ];
    if full_plan {
        keys.push("plan_commitment_sha256");
    }
    require_object_keys(&value, &keys, "contract plan")?;
    Ok(())
}

fn validate_report_json_shape(bytes: &[u8], full_report: bool) -> Result<(), String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("contract report JSON is malformed: {error}"))?;
    let mut keys = vec![
        "schema",
        "purpose",
        "plan_sha256",
        "plan_commitment_sha256",
        "evaluator_source_revision",
        "decision_rules_sha256",
        "adjudication_mode",
        "blind_to_system_identity",
        "records",
    ];
    if full_report {
        keys.push("report_commitment_sha256");
    }
    require_object_keys(&value, &keys, "contract report")?;
    let records = value["records"]
        .as_array()
        .ok_or_else(|| "contract report records must be an array".to_owned())?;
    const RECORD_KEYS: &[&str] = &[
        "case_id",
        "status",
        "baseline_response_artifact_sha256",
        "intervention_response_artifact_sha256",
        "adjudication_record_sha256",
        "violation_codes",
        "not_evaluable_reason",
    ];
    for (index, record) in records.iter().enumerate() {
        require_object_keys(record, RECORD_KEYS, &format!("contract record {index}"))?;
    }
    Ok(())
}

/// Compute a typed commitment from a plan-fields JSON document that omits
/// plan_commitment_sha256. The typed serialization order is part of v1.
pub fn compute_evidence_behavior_contract_plan_commitment(
    fields_bytes: &[u8],
) -> Result<String, String> {
    validate_plan_json_shape(fields_bytes, false)?;
    let fields: EvidenceBehaviorContractPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("contract plan fields are invalid: {error}"))?;
    let issues = validate_plan_fields(&fields);
    if !issues.is_empty() {
        return Err(issues.join("; "));
    }
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize contract plan commitment: {error}"))?;
    digest_domain_framed(PLAN_DOMAIN, &payload)
}

/// Verify a full plan. Structural failures and commitment mismatch are returned
/// as a machine-readable invalid report where possible.
pub fn verify_evidence_behavior_contract_plan(
    plan_bytes: &[u8],
) -> Result<EvidenceBehaviorContractPlanVerificationV1, String> {
    validate_plan_json_shape(plan_bytes, true)?;
    let plan: EvidenceBehaviorContractPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("contract plan is invalid: {error}"))?;
    let fields = fields_from_plan(&plan);
    let mut issues = validate_plan_fields(&fields);
    let recomputed = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize plan commitment payload: {error}"))
        .and_then(|payload| digest_domain_framed(PLAN_DOMAIN, &payload))?;
    let matches = plan.plan_commitment_sha256 == recomputed
        && valid_digest(&plan.plan_commitment_sha256, !plan.purpose.is_qualified());
    if !matches {
        issues.push("plan commitment does not match typed plan fields".to_owned());
    }
    let structural_valid = issues.iter().all(|issue| {
        issue != "plan commitment does not match typed plan fields"
    });
    Ok(EvidenceBehaviorContractPlanVerificationV1 {
        schema: "symthaea.science_bench_refresh_evidence_behavior_contract_plan_verification.v1"
            .to_owned(),
        plan_sha256: hex_digest(&Sha256::digest(plan_bytes)),
        plan_commitment_expected: plan.plan_commitment_sha256.clone(),
        plan_commitment_recomputed: recomputed,
        plan_commitment_matches: matches,
        structural_valid,
        valid: structural_valid && matches && plan.purpose.is_qualified(),
        contract_set_id: plan.contract_set_id.clone(),
        planned_case_count: plan.items.len() as u64,
        issues,
        scoring_eligible_declared: plan.purpose.is_qualified(),
        external_anchor_required: true,
    })
}

fn report_fields_from_report(
    report: &EvidenceBehaviorContractReportV1,
) -> EvidenceBehaviorContractReportFieldsV1 {
    EvidenceBehaviorContractReportFieldsV1 {
        schema: report.schema.clone(),
        purpose: report.purpose,
        plan_sha256: report.plan_sha256.clone(),
        plan_commitment_sha256: report.plan_commitment_sha256.clone(),
        evaluator_source_revision: report.evaluator_source_revision.clone(),
        decision_rules_sha256: report.decision_rules_sha256.clone(),
        adjudication_mode: report.adjudication_mode,
        blind_to_system_identity: report.blind_to_system_identity,
        records: report.records.clone(),
    }
}

fn validate_report_fields(
    plan: &EvidenceBehaviorContractPlanV1,
    plan_bytes: &[u8],
    fields: &EvidenceBehaviorContractReportFieldsV1,
) -> Result<Vec<String>, String> {
    let plan_verification = verify_evidence_behavior_contract_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err(format!(
            "referenced contract plan is not scoring-valid: {}",
            plan_verification.issues.join("; ")
        ));
    }
    let mut issues = Vec::new();
    if fields.schema != EVIDENCE_BEHAVIOR_CONTRACT_REPORT_SCHEMA {
        issues.push("unsupported evidence behavior contract report schema".to_owned());
    }
    if fields.purpose != plan.purpose {
        issues.push("report purpose does not match plan purpose".to_owned());
    }
    let expected_plan_sha = hex_digest(&Sha256::digest(plan_bytes));
    if fields.plan_sha256 != expected_plan_sha {
        issues.push("report does not bind the exact plan bytes".to_owned());
    }
    if fields.plan_commitment_sha256 != plan.plan_commitment_sha256 {
        issues.push("report plan commitment does not match plan".to_owned());
    }
    if fields.evaluator_source_revision != plan.evaluator_source_revision {
        issues.push("report evaluator revision does not match plan".to_owned());
    }
    if fields.decision_rules_sha256 != plan.decision_rules_sha256 {
        issues.push("report decision-rule digest does not match plan".to_owned());
    }
    if fields.purpose.is_qualified() && !fields.blind_to_system_identity {
        issues.push("qualified report must declare identity-blinded adjudication".to_owned());
    }
    if fields.records.len() != plan.items.len() {
        issues.push(format!(
            "record count {} differs from planned case count {}",
            fields.records.len(),
            plan.items.len()
        ));
    }

    let allow_zero = !fields.purpose.is_qualified();
    let planned: BTreeMap<&str, &EvidenceBehaviorContractItemV1> = plan
        .items
        .iter()
        .map(|item| (item.case_id.as_str(), item))
        .collect();
    let mut seen = BTreeSet::new();
    for record in &fields.records {
        if !seen.insert(record.case_id.as_str()) {
            issues.push(format!("duplicate outcome record for case {}", record.case_id));
            continue;
        }
        if !planned.contains_key(record.case_id.as_str()) {
            issues.push(format!("outcome contains unplanned case {}", record.case_id));
        }
        if !valid_digest(&record.adjudication_record_sha256, allow_zero) {
            issues.push(format!(
                "case {} has invalid adjudication_record_sha256",
                record.case_id
            ));
        }
        match record.status {
            ContractOutcomeStatusV1::Pass => {
                if !record.violation_codes.is_empty() || record.not_evaluable_reason.is_some() {
                    issues.push(format!(
                        "passing case {} must have no violation codes or not-evaluable reason",
                        record.case_id
                    ));
                }
                validate_response_digest(
                    &record.case_id,
                    "baseline",
                    &record.baseline_response_artifact_sha256,
                    allow_zero,
                    true,
                    &mut issues,
                );
                validate_response_digest(
                    &record.case_id,
                    "intervention",
                    &record.intervention_response_artifact_sha256,
                    allow_zero,
                    true,
                    &mut issues,
                );
            }
            ContractOutcomeStatusV1::Violation => {
                if record.violation_codes.is_empty() || record.not_evaluable_reason.is_some() {
                    issues.push(format!(
                        "violating case {} requires violation codes and no not-evaluable reason",
                        record.case_id
                    ));
                }
                let mut violation_codes = BTreeSet::new();
                for code in &record.violation_codes {
                    if code.trim().is_empty() || !violation_codes.insert(code) {
                        issues.push(format!(
                            "case {} has empty or duplicate violation code",
                            record.case_id
                        ));
                    }
                }
                validate_response_digest(
                    &record.case_id,
                    "baseline",
                    &record.baseline_response_artifact_sha256,
                    allow_zero,
                    true,
                    &mut issues,
                );
                validate_response_digest(
                    &record.case_id,
                    "intervention",
                    &record.intervention_response_artifact_sha256,
                    allow_zero,
                    true,
                    &mut issues,
                );
            }
            ContractOutcomeStatusV1::NotEvaluable => {
                if !record.violation_codes.is_empty()
                    || record
                        .not_evaluable_reason
                        .as_ref()
                        .is_none_or(|reason| reason.trim().is_empty())
                {
                    issues.push(format!(
                        "not-evaluable case {} requires a reason and no violation codes",
                        record.case_id
                    ));
                }
                validate_response_digest(
                    &record.case_id,
                    "baseline",
                    &record.baseline_response_artifact_sha256,
                    allow_zero,
                    false,
                    &mut issues,
                );
                validate_response_digest(
                    &record.case_id,
                    "intervention",
                    &record.intervention_response_artifact_sha256,
                    allow_zero,
                    false,
                    &mut issues,
                );
            }
        }
    }
    for item in &plan.items {
        if !seen.contains(item.case_id.as_str()) {
            issues.push(format!("missing terminal outcome for case {}", item.case_id));
        }
    }
    Ok(issues)
}

fn validate_response_digest(
    case_id: &str,
    label: &str,
    digest: &Option<String>,
    allow_zero: bool,
    required: bool,
    issues: &mut Vec<String>,
) {
    match digest {
        Some(value) if valid_digest(value, allow_zero) => {}
        Some(_) => issues.push(format!("case {case_id} has invalid {label} response digest")),
        None if required => {
            issues.push(format!("case {case_id} is missing {label} response artifact"))
        }
        None => {}
    }
}

fn report_commitment(
    fields: &EvidenceBehaviorContractReportFieldsV1,
) -> Result<String, String> {
    let payload = OutcomeCommitmentPayloadV1 {
        schema: &fields.schema,
        purpose: fields.purpose,
        plan_sha256: &fields.plan_sha256,
        plan_commitment_sha256: &fields.plan_commitment_sha256,
        evaluator_source_revision: &fields.evaluator_source_revision,
        decision_rules_sha256: &fields.decision_rules_sha256,
        adjudication_mode: fields.adjudication_mode,
        blind_to_system_identity: fields.blind_to_system_identity,
        records: &fields.records,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize contract report commitment: {error}"))?;
    digest_domain_framed(REPORT_DOMAIN, &bytes)
}

/// Compute a report commitment from report-fields JSON without
/// report_commitment_sha256. The referenced plan must already verify.
pub fn compute_evidence_behavior_contract_report_commitment(
    plan_bytes: &[u8],
    fields_bytes: &[u8],
) -> Result<String, String> {
    validate_report_json_shape(fields_bytes, false)?;
    let plan_verification = verify_evidence_behavior_contract_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err("cannot commit outcomes against an invalid contract plan".to_owned());
    }
    let plan: EvidenceBehaviorContractPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("contract plan is invalid: {error}"))?;
    let fields: EvidenceBehaviorContractReportFieldsV1 =
        serde_json::from_slice(fields_bytes)
            .map_err(|error| format!("contract report fields are invalid: {error}"))?;
    let issues = validate_report_fields(&plan, plan_bytes, &fields)?;
    if !issues.is_empty() {
        return Err(issues.join("; "));
    }
    report_commitment(&fields)
}

fn family_scorecard(
    plan: &EvidenceBehaviorContractPlanV1,
    report: &EvidenceBehaviorContractReportV1,
) -> Vec<ContractFamilySummaryV1> {
    #[derive(Default)]
    struct Counts {
        planned: u64,
        pass: u64,
        violations: u64,
        not_evaluable: u64,
    }
    let records: BTreeMap<&str, &EvidenceBehaviorContractOutcomeRecordV1> = report
        .records
        .iter()
        .map(|record| (record.case_id.as_str(), record))
        .collect();
    let mut counts: BTreeMap<ContractFamilyV1, Counts> = BTreeMap::new();
    for item in &plan.items {
        let entry = counts.entry(item.family).or_default();
        entry.planned += 1;
        if let Some(record) = records.get(item.case_id.as_str()) {
            match record.status {
                ContractOutcomeStatusV1::Pass => entry.pass += 1,
                ContractOutcomeStatusV1::Violation => entry.violations += 1,
                ContractOutcomeStatusV1::NotEvaluable => entry.not_evaluable += 1,
            }
        }
    }
    counts
        .into_iter()
        .map(|(family, value)| {
            let eligible = value.pass + value.violations;
            let rate = if eligible == 0 {
                None
            } else {
                Some(((u128::from(value.violations) * 1_000_000 + u128::from(eligible) / 2) / u128::from(eligible)) as u32)
            };
            ContractFamilySummaryV1 {
                family,
                planned_cases: value.planned,
                pass_count: value.pass,
                violation_count: value.violations,
                not_evaluable_count: value.not_evaluable,
                eligible_count: eligible,
                violation_rate_millionths: rate,
            }
        })
        .collect()
}

/// Verify a full terminal report. A valid report can contain contract violations:
/// report validity is about provenance/accounting, not whether the policy passed.
pub fn verify_evidence_behavior_contract_report(
    plan_bytes: &[u8],
    report_bytes: &[u8],
) -> Result<EvidenceBehaviorContractReportVerificationV1, String> {
    validate_report_json_shape(report_bytes, true)?;
    let plan_verification = verify_evidence_behavior_contract_plan(plan_bytes)?;
    if !plan_verification.valid {
        return Err("cannot verify a report against an invalid contract plan".to_owned());
    }
    let plan: EvidenceBehaviorContractPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("contract plan is invalid: {error}"))?;
    let report: EvidenceBehaviorContractReportV1 = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("contract report is invalid: {error}"))?;
    let fields = report_fields_from_report(&report);
    let issues = validate_report_fields(&plan, plan_bytes, &fields)?;
    let recomputed = report_commitment(&fields)?;
    let commitment_matches = report.report_commitment_sha256 == recomputed
        && valid_digest(
            &report.report_commitment_sha256,
            !report.purpose.is_qualified(),
        );
    let mut final_issues = issues;
    if !commitment_matches {
        final_issues.push("report commitment does not match typed report fields".to_owned());
    }
    let missing_count = plan
        .items
        .iter()
        .filter(|item| !report.records.iter().any(|record| record.case_id == item.case_id))
        .count() as u64;
    let valid = final_issues.is_empty()
        && commitment_matches
        && report.purpose.is_qualified();
    let record_counts = report.records.iter().fold((0_u64, 0_u64, 0_u64), |mut acc, r| {
        match r.status {
            ContractOutcomeStatusV1::Pass => acc.0 += 1,
            ContractOutcomeStatusV1::Violation => acc.1 += 1,
            ContractOutcomeStatusV1::NotEvaluable => acc.2 += 1,
        }
        acc
    });
    let scorecard = Some(EvidenceBehaviorContractScorecardV1 {
        schema: "symthaea.science_bench_refresh_evidence_behavior_contract_scorecard.v1"
            .to_owned(),
        plan_sha256: hex_digest(&Sha256::digest(plan_bytes)),
        plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
        report_sha256: hex_digest(&Sha256::digest(report_bytes)),
        report_commitment_sha256: report.report_commitment_sha256.clone(),
        planned_case_count: plan.items.len() as u64,
        observed_record_count: report.records.len() as u64,
        missing_case_count: missing_count,
        pass_count: record_counts.0,
        violation_count: record_counts.1,
        not_evaluable_count: record_counts.2,
        family_summaries: family_scorecard(&plan, &report),
        fixed_benchmark_only: true,
        generalized_accuracy_estimated: false,
        runner_execution_proven: false,
        external_anchor_required: true,
        claim_boundary: "Adjudication is conditional on the frozen contract, evidence-authority policy and rubric; a digest does not prove runner execution, evaluator independence, truth or generalization.".to_owned(),
    });
    Ok(EvidenceBehaviorContractReportVerificationV1 {
        schema: "symthaea.science_bench_refresh_evidence_behavior_contract_report_verification.v1"
            .to_owned(),
        valid,
        plan_binding_valid: final_issues.iter().all(|issue| {
            !issue.contains("plan")
                && !issue.contains("evaluator revision")
                && !issue.contains("decision-rule")
        }),
        report_commitment_matches: commitment_matches,
        planned_case_count: plan.items.len() as u64,
        observed_record_count: report.records.len() as u64,
        missing_case_count: missing_count,
        issues: final_issues,
        scorecard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(char: char) -> String {
        std::iter::repeat_n(char, 64).collect()
    }

    fn plan_fields() -> EvidenceBehaviorContractPlanFieldsV1 {
        let revision = BUILT_GENERATOR_SOURCE_REVISION.to_owned();
        EvidenceBehaviorContractPlanFieldsV1 {
            schema: EVIDENCE_BEHAVIOR_CONTRACT_PLAN_SCHEMA.to_owned(),
            purpose: ContractPurposeV1::QualifiedRun,
            contract_set_id: "test-contracts-v1".to_owned(),
            evaluator_source_revision: revision,
            source_authority_policy_sha256: digest('a'),
            decision_rules_sha256: digest('b'),
            items: vec![
                EvidenceBehaviorContractItemV1 {
                    case_id: "update-1".to_owned(),
                    pair_id: "pair-1".to_owned(),
                    family: ContractFamilyV1::UpdateCompliance,
                    intervention: ContractInterventionV1::CoherentUpdate,
                    baseline_condition: "old".to_owned(),
                    intervention_condition: "new".to_owned(),
                    expected_behavior:
                        ExpectedBehaviorV1::DependentClaimsUpdateWithoutUnrelatedDrift,
                    force_axis: ForceAxisV1::NotApplicable,
                    decision_rule_id: "update-rule-v1".to_owned(),
                    evidence_manifest_sha256: digest('c'),
                    evaluator_gold_sha256: digest('d'),
                },
                EvidenceBehaviorContractItemV1 {
                    case_id: "force-1".to_owned(),
                    pair_id: "pair-2".to_owned(),
                    family: ContractFamilyV1::EvidenceForceMonotonicity,
                    intervention: ContractInterventionV1::EvidenceStrengthReduction,
                    baseline_condition: "strong".to_owned(),
                    intervention_condition: "weak".to_owned(),
                    expected_behavior: ExpectedBehaviorV1::WeakerEvidenceNotStrongerClaim,
                    force_axis: ForceAxisV1::NumericSpecificity,
                    decision_rule_id: "force-rule-v1".to_owned(),
                    evidence_manifest_sha256: digest('e'),
                    evaluator_gold_sha256: digest('f'),
                },
            ],
        }
    }

    fn committed_plan_bytes() -> Vec<u8> {
        let fields = plan_fields();
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_evidence_behavior_contract_plan_commitment(&fields_bytes).unwrap();
        let plan = EvidenceBehaviorContractPlanV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            contract_set_id: fields.contract_set_id,
            evaluator_source_revision: fields.evaluator_source_revision,
            source_authority_policy_sha256: fields.source_authority_policy_sha256,
            decision_rules_sha256: fields.decision_rules_sha256,
            items: fields.items,
            plan_commitment_sha256: commitment,
        };
        serde_json::to_vec(&plan).unwrap()
    }

    fn outcome_fields(plan_bytes: &[u8]) -> EvidenceBehaviorContractReportFieldsV1 {
        let plan: EvidenceBehaviorContractPlanV1 = serde_json::from_slice(plan_bytes).unwrap();
        let response_a = digest('1');
        let response_b = digest('2');
        EvidenceBehaviorContractReportFieldsV1 {
            schema: EVIDENCE_BEHAVIOR_CONTRACT_REPORT_SCHEMA.to_owned(),
            purpose: ContractPurposeV1::QualifiedRun,
            plan_sha256: hex_digest(&Sha256::digest(plan_bytes)),
            plan_commitment_sha256: plan.plan_commitment_sha256,
            evaluator_source_revision: plan.evaluator_source_revision,
            decision_rules_sha256: plan.decision_rules_sha256,
            adjudication_mode: ContractAdjudicationModeV1::BlindedHuman,
            blind_to_system_identity: true,
            records: vec![
                EvidenceBehaviorContractOutcomeRecordV1 {
                    case_id: "update-1".to_owned(),
                    status: ContractOutcomeStatusV1::Pass,
                    baseline_response_artifact_sha256: Some(response_a.clone()),
                    intervention_response_artifact_sha256: Some(response_b.clone()),
                    adjudication_record_sha256: digest('3'),
                    violation_codes: vec![],
                    not_evaluable_reason: None,
                },
                EvidenceBehaviorContractOutcomeRecordV1 {
                    case_id: "force-1".to_owned(),
                    status: ContractOutcomeStatusV1::Violation,
                    baseline_response_artifact_sha256: Some(response_a),
                    intervention_response_artifact_sha256: Some(response_b),
                    adjudication_record_sha256: digest('4'),
                    violation_codes: vec!["unsupported_force_increase".to_owned()],
                    not_evaluable_reason: None,
                },
            ],
        }
    }

    #[test]
    fn committed_plan_verifies_at_exact_build_revision() {
        let plan_bytes = committed_plan_bytes();
        let report = verify_evidence_behavior_contract_plan(&plan_bytes).unwrap();
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.planned_case_count, 2);
        assert!(report.external_anchor_required);
    }

    #[test]
    fn plan_rejects_family_behavior_mismatch() {
        let mut fields = plan_fields();
        fields.items[0].expected_behavior = ExpectedBehaviorV1::SupportedClaimsStable;
        let bytes = serde_json::to_vec(&fields).unwrap();
        assert!(compute_evidence_behavior_contract_plan_commitment(&bytes).is_err());
    }

    #[test]
    fn report_preserves_violation_as_data_not_invalid_receipt() {
        let plan_bytes = committed_plan_bytes();
        let fields = outcome_fields(&plan_bytes);
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_evidence_behavior_contract_report_commitment(
            &plan_bytes,
            &fields_bytes,
        )
        .unwrap();
        let report = EvidenceBehaviorContractReportV1 {
            schema: fields.schema,
            purpose: fields.purpose,
            plan_sha256: fields.plan_sha256,
            plan_commitment_sha256: fields.plan_commitment_sha256,
            evaluator_source_revision: fields.evaluator_source_revision,
            decision_rules_sha256: fields.decision_rules_sha256,
            adjudication_mode: fields.adjudication_mode,
            blind_to_system_identity: fields.blind_to_system_identity,
            records: fields.records,
            report_commitment_sha256: commitment,
        };
        let report_bytes = serde_json::to_vec(&report).unwrap();
        let verified =
            verify_evidence_behavior_contract_report(&plan_bytes, &report_bytes).unwrap();
        assert!(verified.valid, "{:?}", verified.issues);
        let scorecard = verified.scorecard.unwrap();
        assert_eq!(scorecard.pass_count, 1);
        assert_eq!(scorecard.violation_count, 1);
        assert!(!scorecard.runner_execution_proven);
        assert!(scorecard.external_anchor_required);
    }

    #[test]
    fn report_rejects_duplicate_case_and_missing_case() {
        let plan_bytes = committed_plan_bytes();
        let mut fields = outcome_fields(&plan_bytes);
        fields.records[1].case_id = fields.records[0].case_id.clone();
        assert!(
            compute_evidence_behavior_contract_report_commitment(
                &plan_bytes,
                &serde_json::to_vec(&fields).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn violation_requires_a_violation_code() {
        let plan_bytes = committed_plan_bytes();
        let mut fields = outcome_fields(&plan_bytes);
        fields.records[1].violation_codes.clear();
        assert!(
            compute_evidence_behavior_contract_report_commitment(
                &plan_bytes,
                &serde_json::to_vec(&fields).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn plan_bytes_are_bound_exactly_in_report() {
        let plan_bytes = committed_plan_bytes();
        let fields = outcome_fields(&plan_bytes);
        let mut changed_plan_bytes = plan_bytes.clone();
        changed_plan_bytes.push(b'\n');
        assert!(
            compute_evidence_behavior_contract_report_commitment(
                &changed_plan_bytes,
                &serde_json::to_vec(&fields).unwrap()
            )
            .is_err()
        );
    }
}
