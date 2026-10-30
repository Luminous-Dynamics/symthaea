//! Finite-sample, calibration-aware certification with an imperfect binary judge.
//!
//! The analyzer uses independent human-adjudicated calibration items to derive
//! one-sided Hoeffding bounds for judge sensitivity, false-positive rate, and
//! test-set judge-positive prevalence. It inverts the worst-case binary
//! measurement channel to upper-bound true failure prevalence.
//!
//! Guarantees are conditional on the declared iid-within-stratum sampling and
//! calibration/test exchangeability assumptions. The analyzer does not prove
//! those assumptions, human-label truth, execution authenticity, or population
//! representativeness.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CERTIFICATION_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_imperfect_judge_certification_plan.v1";
pub const CERTIFICATION_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_imperfect_judge_certification_receipt.v1";
pub const CERTIFICATION_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_imperfect_judge_certification_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.imperfect_judge_certification.plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.imperfect_judge_certification.report.v1\0";
const MAX_STRATA: usize = 128;
const MAX_CALIBRATION_ITEMS: usize = 100_000;
const MAX_TEST_ITEMS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CertificationPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl CertificationPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HumanReferenceLabelV1 {
    Failure,
    NonFailure,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JudgeOutcomeV1 {
    Failure,
    NonFailure,
    Abstain,
}

impl JudgeOutcomeV1 {
    /// Safety-conservative binary decision: abstention is counted as failure-positive.
    fn is_positive(self) -> bool {
        matches!(self, Self::Failure | Self::Abstain)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FamilyCertificationStatusV1 {
    CertifiedBelowThreshold,
    ThresholdExceeded,
    JudgeUninformative,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OverallCertificationStatusV1 {
    CertifiedBelowThreshold,
    ThresholdExceeded,
    JudgeUninformative,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CertificationStratumV1 {
    pub task_family: String,
    pub minimum_calibration_failure_items: u32,
    pub minimum_calibration_nonfailure_items: u32,
    pub minimum_test_items: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationItemPlanV1 {
    pub item_id: String,
    pub task_family: String,
    pub input_sha256: String,
    pub reference_label: HumanReferenceLabelV1,
    pub reference_adjudication_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TestItemPlanV1 {
    pub item_id: String,
    pub task_family: String,
    pub input_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CertificationPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CertificationPurposeV1,
    pub study_id: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub human_reference_protocol_sha256: String,
    pub sampling_frame_sha256: String,
    pub target_failure_rate_millionths: u32,
    pub familywise_alpha_millionths: u32,
    pub strata: Vec<CertificationStratumV1>,
    pub calibration_items: Vec<CalibrationItemPlanV1>,
    pub test_items: Vec<TestItemPlanV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CertificationPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CertificationPurposeV1,
    pub study_id: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub human_reference_protocol_sha256: String,
    pub sampling_frame_sha256: String,
    pub target_failure_rate_millionths: u32,
    pub familywise_alpha_millionths: u32,
    pub strata: Vec<CertificationStratumV1>,
    pub calibration_items: Vec<CalibrationItemPlanV1>,
    pub test_items: Vec<TestItemPlanV1>,
    pub plan_commitment_sha256: String,
}

impl CertificationPlanV1 {
    fn fields(&self) -> CertificationPlanFieldsV1 {
        CertificationPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            study_id: self.study_id.clone(),
            judge_source_revision: self.judge_source_revision.clone(),
            judge_prompt_sha256: self.judge_prompt_sha256.clone(),
            rubric_sha256: self.rubric_sha256.clone(),
            human_reference_protocol_sha256: self.human_reference_protocol_sha256.clone(),
            sampling_frame_sha256: self.sampling_frame_sha256.clone(),
            target_failure_rate_millionths: self.target_failure_rate_millionths,
            familywise_alpha_millionths: self.familywise_alpha_millionths,
            strata: self.strata.clone(),
            calibration_items: self.calibration_items.clone(),
            test_items: self.test_items.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationObservationV1 {
    pub item_id: String,
    pub input_sha256: String,
    pub judge_outcome: JudgeOutcomeV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TestObservationV1 {
    pub item_id: String,
    pub input_sha256: String,
    pub judge_outcome: JudgeOutcomeV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CertificationReceiptV1 {
    pub schema: String,
    pub purpose: CertificationPurposeV1,
    pub run_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub analyzer_source_revision: String,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub human_reference_protocol_sha256: String,
    pub calibration_observations: Vec<CalibrationObservationV1>,
    pub test_observations: Vec<TestObservationV1>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FamilyCertificationResultV1 {
    pub task_family: String,
    pub calibration_failure_count: u64,
    pub calibration_nonfailure_count: u64,
    pub test_item_count: u64,
    pub true_positive_count: u64,
    pub false_negative_count: u64,
    pub false_positive_count: u64,
    pub true_negative_count: u64,
    pub calibration_failure_abstention_count: u64,
    pub calibration_nonfailure_abstention_count: u64,
    pub test_abstention_count: u64,
    pub sensitivity_lower_millionths: u32,
    pub false_positive_rate_lower_millionths: u32,
    pub judge_positive_rate_upper_millionths: u32,
    pub true_failure_rate_upper_millionths: Option<u32>,
    pub alpha_per_bound_millionths: u32,
    pub hoeffding_radius_failure_calibration_millionths: u32,
    pub hoeffding_radius_nonfailure_calibration_millionths: u32,
    pub hoeffding_radius_test_millionths: u32,
    pub status: FamilyCertificationStatusV1,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CertificationReportFieldsV1 {
    pub schema: String,
    pub study_id: String,
    pub analyzer_source_revision: String,
    pub analyzer_worktree_clean: bool,
    pub judge_source_revision: String,
    pub judge_prompt_sha256: String,
    pub rubric_sha256: String,
    pub human_reference_protocol_sha256: String,
    pub sampling_frame_sha256: String,
    pub plan_sha256: String,
    pub receipt_sha256: String,
    pub plan_commitment_sha256: String,
    pub report_commitment_sha256_placeholder: Option<String>,
    pub overall_status: OverallCertificationStatusV1,
    pub target_failure_rate_millionths: u32,
    pub familywise_alpha_millionths: u32,
    pub simultaneous_confidence_lower_millionths: u32,
    pub stratum_count: u64,
    pub familywise_bound_count: u64,
    pub planned_calibration_item_count: u64,
    pub observed_calibration_item_count: u64,
    pub planned_test_item_count: u64,
    pub observed_test_item_count: u64,
    pub abstentions_count_as_failure_positive: bool,
    pub bound_method: String,
    pub calibration_test_ids_disjoint: bool,
    pub conditional_iid_within_stratum_assumption: bool,
    pub calibration_test_exchangeability_assumption: bool,
    pub assumptions_empirically_verified: bool,
    pub human_reference_truth_proven: bool,
    pub execution_authenticity_proven: bool,
    pub population_representativeness_proven: bool,
    pub causal_effect_identified: bool,
    pub generalized_accuracy_estimated: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub family_results: Vec<FamilyCertificationResultV1>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CertificationReportV1 {
    #[serde(flatten)]
    pub fields: CertificationReportFieldsV1,
    pub report_commitment_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn domain_digest(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let length = u64::try_from(payload.len())
        .map_err(|_| "domain-separated payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(length.to_be_bytes());
    hasher.update(payload);
    Ok(sha256_hex(&hasher.finalize()))
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value.bytes().any(|byte| byte != b'0'))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn nonempty(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
}

fn plan_fields_commitment(fields: &CertificationPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize certification plan fields: {error}"))?;
    domain_digest(PLAN_DOMAIN, &payload)
}

fn validate_plan_fields(fields: &CertificationPlanFieldsV1) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();
    if fields.schema != CERTIFICATION_PLAN_SCHEMA {
        issues.push("unsupported imperfect-judge certification plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push("analyzer_source_revision must match the exact compiled validator revision".to_owned());
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues.push("qualified certification plans require a clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.study_id, 128) {
        issues.push("study_id must be non-empty and bounded".to_owned());
    }
    if !valid_revision(&fields.judge_source_revision) {
        issues.push("judge_source_revision must be a nonzero 40-character lowercase revision".to_owned());
    }
    for (label, value) in [
        ("judge_prompt_sha256", &fields.judge_prompt_sha256),
        ("rubric_sha256", &fields.rubric_sha256),
        ("human_reference_protocol_sha256", &fields.human_reference_protocol_sha256),
        ("sampling_frame_sha256", &fields.sampling_frame_sha256),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!("{label} must be a valid nonzero SHA-256 digest"));
        }
    }
    if fields.target_failure_rate_millionths >= 1_000_000 {
        issues.push("target_failure_rate_millionths must be below 1,000,000".to_owned());
    }
    if fields.familywise_alpha_millionths == 0 || fields.familywise_alpha_millionths >= 500_000 {
        issues.push("familywise_alpha_millionths must be in 1..500,000".to_owned());
    }
    if fields.strata.is_empty() || fields.strata.len() > MAX_STRATA {
        issues.push(format!("strata must contain 1..={MAX_STRATA} rows"));
    }
    if fields.calibration_items.is_empty() || fields.calibration_items.len() > MAX_CALIBRATION_ITEMS {
        issues.push(format!("calibration_items must contain 1..={MAX_CALIBRATION_ITEMS} rows"));
    }
    if fields.test_items.is_empty() || fields.test_items.len() > MAX_TEST_ITEMS {
        issues.push(format!("test_items must contain 1..={MAX_TEST_ITEMS} rows"));
    }

    let mut strata = BTreeMap::new();
    for stratum in &fields.strata {
        if !nonempty(&stratum.task_family, 128) || strata.contains_key(stratum.task_family.as_str()) {
            issues.push("task-family strata must have unique non-empty bounded identifiers".to_owned());
        }
        if stratum.minimum_calibration_failure_items < 2
            || stratum.minimum_calibration_nonfailure_items < 2
            || stratum.minimum_test_items < 2
        {
            issues.push(format!("stratum {} requires at least two calibration positive, calibration negative and test items", stratum.task_family));
        }
        strata.insert(
            stratum.task_family.as_str(),
            (
                stratum.minimum_calibration_failure_items,
                stratum.minimum_calibration_nonfailure_items,
                stratum.minimum_test_items,
            ),
        );
    }

    let mut all_ids = BTreeSet::new();
    let mut cal_counts: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
    for item in &fields.calibration_items {
        if !nonempty(&item.item_id, 128) || !all_ids.insert(item.item_id.as_str()) {
            issues.push("calibration item IDs must be unique across all plan samples".to_owned());
        }
        if !valid_digest(&item.input_sha256, allow_zero)
            || !valid_digest(&item.reference_adjudication_sha256, allow_zero)
        {
            issues.push(format!("calibration item {} has invalid input/adjudication digest", item.item_id));
        }
        if let Some((positive, negative)) = cal_counts.get_mut(item.task_family.as_str()) {
            match item.reference_label {
                HumanReferenceLabelV1::Failure => *positive += 1,
                HumanReferenceLabelV1::NonFailure => *negative += 1,
            }
        } else {
            issues.push(format!("calibration item {} references an undeclared task family", item.item_id));
        }
    }

    let mut test_counts: BTreeMap<&str, u32> = BTreeMap::new();
    for item in &fields.test_items {
        if !nonempty(&item.item_id, 128) || !all_ids.insert(item.item_id.as_str()) {
            issues.push("calibration and test item IDs must be globally unique and disjoint".to_owned());
        }
        if !valid_digest(&item.input_sha256, allow_zero) {
            issues.push(format!("test item {} has invalid input_sha256", item.item_id));
        }
        if let Some(count) = test_counts.get_mut(item.task_family.as_str()) {
            *count += 1;
        } else {
            issues.push(format!("test item {} references an undeclared task family", item.item_id));
        }
    }

    for (family, (min_positive, min_negative, min_test)) in &strata {
        let (positive, negative) = cal_counts.get(family).copied().unwrap_or((0, 0));
        let test = test_counts.get(family).copied().unwrap_or(0);
        if positive < *min_positive || negative < *min_negative || test < *min_test {
            issues.push(format!(
                "stratum {family} planned calibration/test counts are below the preregistered minima"
            ));
        }
    }
    if all_ids.len() != fields.calibration_items.len() + fields.test_items.len() {
        issues.push("calibration/test ID disjointness failure".to_owned());
    }
    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

pub fn compute_certification_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: CertificationPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("certification plan fields invalid: {error}"))?;
    validate_plan_fields(&fields).map_err(|issues| issues.join("; "))?;
    plan_fields_commitment(&fields)
}

pub fn verify_certification_plan(plan_bytes: &[u8]) -> Result<(CertificationPlanV1, String), String> {
    let plan: CertificationPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("certification plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_plan_fields(&fields).map_err(|issues| issues.join("; "))?;
    let expected = plan_fields_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("imperfect-judge certification plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

fn hoeffding_radius_millionths(n: u64, alpha: f64) -> u32 {
    let radius = (alpha.recip().ln() / (2.0 * n as f64)).sqrt();
    (radius.mul_add(1_000_000.0, 0.999_999)).floor().clamp(0.0, 1_000_000.0) as u32
}

fn lower_bound_millionths(successes: u64, total: u64, radius: u32) -> u32 {
    let estimate = successes as f64 / total as f64;
    ((estimate * 1_000_000.0 - f64::from(radius)).floor())
        .clamp(0.0, 1_000_000.0) as u32
}

fn upper_bound_millionths(successes: u64, total: u64, radius: u32) -> u32 {
    let estimate = successes as f64 / total as f64;
    ((estimate * 1_000_000.0 + f64::from(radius)).ceil())
        .clamp(0.0, 1_000_000.0) as u32
}

fn invert_judge_channel(
    judge_positive_upper: u32,
    sensitivity_lower: u32,
    false_positive_lower: u32,
) -> Option<u32> {
    if sensitivity_lower <= false_positive_lower {
        return None;
    }
    if judge_positive_upper <= false_positive_lower {
        return Some(0);
    }
    let numerator = f64::from(judge_positive_upper - false_positive_lower);
    let denominator = f64::from(sensitivity_lower - false_positive_lower);
    Some((numerator / denominator * 1_000_000.0).ceil().clamp(0.0, 1_000_000.0) as u32)
}

pub fn analyze_imperfect_judge_certification(
    plan_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<CertificationReportV1, String> {
    let (plan, plan_sha256) = verify_certification_plan(plan_bytes)?;
    let receipt: CertificationReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("certification receipt invalid: {error}"))?;
    if receipt.schema != CERTIFICATION_RECEIPT_SCHEMA {
        return Err("unsupported imperfect-judge certification receipt schema".to_owned());
    }
    if receipt.purpose != plan.purpose {
        return Err("receipt purpose differs from plan".to_owned());
    }
    if !nonempty(&receipt.run_id, 128) {
        return Err("receipt run_id must be non-empty and bounded".to_owned());
    }
    if receipt.analyzer_source_revision != plan.analyzer_source_revision
        || receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        return Err("receipt analyzer revision differs from exact compiled validator revision".to_owned());
    }
    if receipt.judge_source_revision != plan.judge_source_revision
        || receipt.judge_prompt_sha256 != plan.judge_prompt_sha256
        || receipt.rubric_sha256 != plan.rubric_sha256
        || receipt.human_reference_protocol_sha256 != plan.human_reference_protocol_sha256
    {
        return Err("receipt judge/reference protocol differs from frozen plan".to_owned());
    }
    if receipt.plan_sha256 != plan_sha256
        || receipt.plan_commitment_sha256 != plan.plan_commitment_sha256
    {
        return Err("receipt does not bind exact plan bytes and commitment".to_owned());
    }

    let calibration_plan: BTreeMap<&str, &CalibrationItemPlanV1> = plan.calibration_items.iter()
        .map(|item| (item.item_id.as_str(), item)).collect();
    let test_plan: BTreeMap<&str, &TestItemPlanV1> = plan.test_items.iter()
        .map(|item| (item.item_id.as_str(), item)).collect();
    let mut calibration_observations: BTreeMap<&str, &CalibrationObservationV1> = BTreeMap::new();
    for observation in &receipt.calibration_observations {
        if calibration_observations.insert(observation.item_id.as_str(), observation).is_some() {
            return Err(format!("duplicate calibration observation {}", observation.item_id));
        }
        let item = calibration_plan.get(observation.item_id.as_str())
            .ok_or_else(|| format!("unexpected calibration item {}", observation.item_id))?;
        if observation.input_sha256 != item.input_sha256 {
            return Err(format!("calibration input digest differs for {}", observation.item_id));
        }
    }
    if calibration_observations.len() != plan.calibration_items.len() {
        return Err("receipt is missing planned calibration observations".to_owned());
    }
    let mut test_observations: BTreeMap<&str, &TestObservationV1> = BTreeMap::new();
    for observation in &receipt.test_observations {
        if test_observations.insert(observation.item_id.as_str(), observation).is_some() {
            return Err(format!("duplicate test observation {}", observation.item_id));
        }
        let item = test_plan.get(observation.item_id.as_str())
            .ok_or_else(|| format!("unexpected test item {}", observation.item_id))?;
        if observation.input_sha256 != item.input_sha256 {
            return Err(format!("test input digest differs for {}", observation.item_id));
        }
    }
    if test_observations.len() != plan.test_items.len() {
        return Err("receipt is missing planned test observations".to_owned());
    }

    let alpha = f64::from(plan.familywise_alpha_millionths) / 1_000_000.0;
    let alpha_per_bound = alpha / (3.0 * plan.strata.len() as f64);
    let alpha_per_bound_millionths = (alpha_per_bound * 1_000_000.0).floor().max(1.0) as u32;
    let mut family_results = Vec::new();

    for stratum in &plan.strata {
        let calibration_items: Vec<&CalibrationItemPlanV1> = plan.calibration_items.iter()
            .filter(|item| item.task_family == stratum.task_family).collect();
        let test_items: Vec<&TestItemPlanV1> = plan.test_items.iter()
            .filter(|item| item.task_family == stratum.task_family).collect();
        let mut tp = 0_u64;
        let mut fn_count = 0_u64;
        let mut fp = 0_u64;
        let mut tn = 0_u64;
        let mut calibration_failure_abstentions = 0_u64;
        let mut calibration_nonfailure_abstentions = 0_u64;
        let mut test_positive = 0_u64;
        let mut test_abstentions = 0_u64;
        let mut calibration_positive = 0_u64;
        let mut calibration_negative = 0_u64;

        for item in calibration_items {
            let observation = calibration_observations.get(item.item_id.as_str()).unwrap();
            let judge_positive = observation.judge_outcome.is_positive();
            match (item.reference_label, judge_positive) {
                (HumanReferenceLabelV1::Failure, true) => tp += 1,
                (HumanReferenceLabelV1::Failure, false) => fn_count += 1,
                (HumanReferenceLabelV1::NonFailure, true) => fp += 1,
                (HumanReferenceLabelV1::NonFailure, false) => tn += 1,
            }
            match (item.reference_label, observation.judge_outcome) {
                (HumanReferenceLabelV1::Failure, JudgeOutcomeV1::Abstain) => calibration_failure_abstentions += 1,
                (HumanReferenceLabelV1::NonFailure, JudgeOutcomeV1::Abstain) => calibration_nonfailure_abstentions += 1,
                _ => {}
            }
            match item.reference_label {
                HumanReferenceLabelV1::Failure => calibration_positive += 1,
                HumanReferenceLabelV1::NonFailure => calibration_negative += 1,
            }
        }
        for item in test_items {
            let observation = test_observations.get(item.item_id.as_str()).unwrap();
            if observation.judge_outcome.is_positive() {
                test_positive += 1;
            }
            if observation.judge_outcome == JudgeOutcomeV1::Abstain {
                test_abstentions += 1;
            }
        }

        let test_count = test_observations.values()
            .filter(|observation| test_plan[observation.item_id.as_str()].task_family == stratum.task_family)
            .count() as u64;
        let calibration_failure_radius = hoeffding_radius_millionths(calibration_positive, alpha_per_bound);
        let calibration_nonfailure_radius = hoeffding_radius_millionths(calibration_negative, alpha_per_bound);
        let test_radius = hoeffding_radius_millionths(test_count, alpha_per_bound);
        let sensitivity_lower = lower_bound_millionths(tp, calibration_positive, calibration_failure_radius);
        let false_positive_lower = lower_bound_millionths(fp, calibration_negative, calibration_nonfailure_radius);
        let judge_positive_upper = upper_bound_millionths(test_positive, test_count, test_radius);
        let true_failure_upper = invert_judge_channel(
            judge_positive_upper, sensitivity_lower, false_positive_lower,
        );
        let status = match true_failure_upper {
            None => FamilyCertificationStatusV1::JudgeUninformative,
            Some(upper) if upper <= plan.target_failure_rate_millionths => {
                FamilyCertificationStatusV1::CertifiedBelowThreshold
            }
            Some(_) => FamilyCertificationStatusV1::ThresholdExceeded,
        };
        family_results.push(FamilyCertificationResultV1 {
            task_family: stratum.task_family.clone(),
            calibration_failure_count: calibration_positive,
            calibration_nonfailure_count: calibration_negative,
            test_item_count: test_count,
            true_positive_count: tp,
            false_negative_count: fn_count,
            false_positive_count: fp,
            true_negative_count: tn,
            calibration_failure_abstention_count: calibration_failure_abstentions,
            calibration_nonfailure_abstention_count: calibration_nonfailure_abstentions,
            test_abstention_count: test_abstentions,
            sensitivity_lower_millionths: sensitivity_lower,
            false_positive_rate_lower_millionths: false_positive_lower,
            judge_positive_rate_upper_millionths: judge_positive_upper,
            true_failure_rate_upper_millionths: true_failure_upper,
            alpha_per_bound_millionths,
            hoeffding_radius_failure_calibration_millionths: calibration_failure_radius,
            hoeffding_radius_nonfailure_calibration_millionths: calibration_nonfailure_radius,
            hoeffding_radius_test_millionths: test_radius,
            status,
        });
    }

    let overall = if family_results.iter().any(|row| row.status == FamilyCertificationStatusV1::JudgeUninformative) {
        OverallCertificationStatusV1::JudgeUninformative
    } else if family_results.iter().any(|row| row.status == FamilyCertificationStatusV1::ThresholdExceeded) {
        OverallCertificationStatusV1::ThresholdExceeded
    } else {
        OverallCertificationStatusV1::CertifiedBelowThreshold
    };
    let calibration_disjoint = plan.calibration_items.iter().all(|item| {
        !test_plan.contains_key(item.item_id.as_str())
    });
    let fields = CertificationReportFieldsV1 {
        schema: CERTIFICATION_REPORT_SCHEMA.to_owned(),
        study_id: plan.study_id,
        analyzer_source_revision: plan.analyzer_source_revision,
        analyzer_worktree_clean: crate::BUILT_GENERATOR_WORKTREE_CLEAN == "true",
        judge_source_revision: plan.judge_source_revision,
        judge_prompt_sha256: plan.judge_prompt_sha256,
        rubric_sha256: plan.rubric_sha256,
        human_reference_protocol_sha256: plan.human_reference_protocol_sha256,
        sampling_frame_sha256: plan.sampling_frame_sha256,
        plan_sha256,
        receipt_sha256: sha256_hex(receipt_bytes),
        plan_commitment_sha256: plan.plan_commitment_sha256,
        report_commitment_sha256_placeholder: None,
        overall_status: overall,
        target_failure_rate_millionths: plan.target_failure_rate_millionths,
        familywise_alpha_millionths: plan.familywise_alpha_millionths,
        simultaneous_confidence_lower_millionths: 1_000_000 - plan.familywise_alpha_millionths,
        stratum_count: plan.strata.len() as u64,
        familywise_bound_count: (plan.strata.len() * 3) as u64,
        planned_calibration_item_count: plan.calibration_items.len() as u64,
        observed_calibration_item_count: calibration_observations.len() as u64,
        planned_test_item_count: plan.test_items.len() as u64,
        observed_test_item_count: test_observations.len() as u64,
        abstentions_count_as_failure_positive: true,
        bound_method: "one_sided_hoeffding_with_bonferroni".to_owned(),
        calibration_test_ids_disjoint: calibration_disjoint,
        conditional_iid_within_stratum_assumption: true,
        calibration_test_exchangeability_assumption: true,
        assumptions_empirically_verified: false,
        human_reference_truth_proven: false,
        execution_authenticity_proven: false,
        population_representativeness_proven: false,
        causal_effect_identified: false,
        generalized_accuracy_estimated: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        family_results,
        claim_boundary: "Finite-sample one-sided Hoeffding/Bonferroni bounds are conditional on independent, identically distributed calibration and test observations within each declared task-family stratum, stable judge operating characteristics between calibration and test, and correct adjudicated reference labels. The analyzer checks data commitments, disjoint item IDs, denominators and calculations but does not prove those assumptions, human-reference truth, execution authenticity, population representativeness or scoring eligibility. Abstentions count as failure-positive outcomes.".to_owned(),
    };
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize certification report fields: {error}"))?;
    let report_commitment = domain_digest(REPORT_DOMAIN, &payload)?;
    Ok(CertificationReportV1 {
        fields,
        report_commitment_sha256: report_commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn make_plan() -> (CertificationPlanV1, Vec<u8>, Vec<u8>) {
        let stratum = CertificationStratumV1 {
            task_family: "safety".to_owned(),
            minimum_calibration_failure_items: 8,
            minimum_calibration_nonfailure_items: 8,
            minimum_test_items: 32,
        };
        let mut calibration_items = Vec::new();
        let mut calibration_observations = Vec::new();
        for index in 0..16_u32 {
            let failure = index < 8;
            let item_id = format!("cal-{index:02}");
            calibration_items.push(CalibrationItemPlanV1 {
                item_id: item_id.clone(),
                task_family: "safety".to_owned(),
                input_sha256: digest(&format!("input-{item_id}")),
                reference_label: if failure { HumanReferenceLabelV1::Failure } else { HumanReferenceLabelV1::NonFailure },
                reference_adjudication_sha256: digest(&format!("adjudication-{item_id}")),
            });
            calibration_observations.push(CalibrationObservationV1 {
                item_id,
                input_sha256: digest(&format!("input-cal-{index:02}")),
                judge_outcome: if index % 4 == 0 {
                    JudgeOutcomeV1::Abstain
                } else {
                    if failure { JudgeOutcomeV1::Failure } else { JudgeOutcomeV1::NonFailure }
                },
            });
        }
        let mut test_items = Vec::new();
        let mut test_observations = Vec::new();
        for index in 0..32_u32 {
            let item_id = format!("test-{index:02}");
            test_items.push(TestItemPlanV1 {
                item_id: item_id.clone(),
                task_family: "safety".to_owned(),
                input_sha256: digest(&format!("input-{item_id}")),
            });
            test_observations.push(TestObservationV1 {
                item_id,
                input_sha256: digest(&format!("input-test-{index:02}")),
                judge_outcome: if index < 5 {
                    JudgeOutcomeV1::Failure
                } else {
                    JudgeOutcomeV1::NonFailure
                },
            });
        }
        let fields = CertificationPlanFieldsV1 {
            schema: CERTIFICATION_PLAN_SCHEMA.to_owned(),
            analyzer_source_revision: crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            purpose: CertificationPurposeV1::QualifiedRun,
            study_id: "certification-synthetic-example".to_owned(),
            judge_source_revision: "a".repeat(40),
            judge_prompt_sha256: digest("prompt"),
            rubric_sha256: digest("rubric"),
            human_reference_protocol_sha256: digest("human protocol"),
            sampling_frame_sha256: digest("sampling frame"),
            target_failure_rate_millionths: 200_000,
            familywise_alpha_millionths: 50_000,
            strata: vec![stratum],
            calibration_items,
            test_items,
        };
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = CertificationPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            study_id: fields.study_id.clone(),
            judge_source_revision: fields.judge_source_revision.clone(),
            judge_prompt_sha256: fields.judge_prompt_sha256.clone(),
            rubric_sha256: fields.rubric_sha256.clone(),
            human_reference_protocol_sha256: fields.human_reference_protocol_sha256.clone(),
            sampling_frame_sha256: fields.sampling_frame_sha256.clone(),
            target_failure_rate_millionths: fields.target_failure_rate_millionths,
            familywise_alpha_millionths: fields.familywise_alpha_millionths,
            strata: fields.strata.clone(),
            calibration_items: fields.calibration_items.clone(),
            test_items: fields.test_items.clone(),
            plan_commitment_sha256: commitment,
        };
        let plan_bytes = serde_json::to_vec_pretty(&plan).unwrap();
        let receipt = CertificationReceiptV1 {
            schema: CERTIFICATION_RECEIPT_SCHEMA.to_owned(),
            purpose: CertificationPurposeV1::QualifiedRun,
            run_id: "run-001".to_owned(),
            plan_sha256: sha256_hex(&plan_bytes),
            plan_commitment_sha256: commitment,
            analyzer_source_revision: plan.analyzer_source_revision.clone(),
            judge_source_revision: plan.judge_source_revision.clone(),
            judge_prompt_sha256: plan.judge_prompt_sha256.clone(),
            rubric_sha256: plan.rubric_sha256.clone(),
            human_reference_protocol_sha256: plan.human_reference_protocol_sha256.clone(),
            calibration_observations,
            test_observations,
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        (plan, plan_bytes, receipt_bytes)
    }

    #[test]
    fn certification_is_deterministic_and_reports_familywise_bound() {
        let (_, plan, receipt) = make_plan();
        let report = analyze_imperfect_judge_certification(&plan, &receipt).unwrap();
        assert_eq!(report.fields.familywise_bound_count, 3);
        assert_eq!(report.fields.simultaneous_confidence_lower_millionths, 950_000);
        assert!(report.fields.abstentions_count_as_failure_positive);
        assert!(!report.fields.assumptions_empirically_verified);
        assert!(!report.fields.human_reference_truth_proven);
        assert!(!report.fields.scoring_eligible_declared);
        let first = serde_json::to_vec(&report).unwrap();
        let second = serde_json::to_vec(&analyze_imperfect_judge_certification(&plan, &receipt).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn abstentions_increase_positive_counts_instead_of_disappearing() {
        let (_, plan, receipt) = make_plan();
        let report = analyze_imperfect_judge_certification(&plan, &receipt).unwrap();
        let family = &report.fields.family_results[0];
        assert_eq!(family.calibration_failure_abstention_count, 2);
        assert_eq!(family.calibration_nonfailure_abstention_count, 2);
        assert_eq!(family.test_abstention_count, 0);
        assert_eq!(family.true_positive_count + family.false_negative_count, 8);
        assert_eq!(family.false_positive_count + family.true_negative_count, 8);
    }

    #[test]
    fn rejects_calibration_and_test_item_id_overlap() {
        let (_, plan, _) = make_plan();
        let mut fields = plan.fields();
        fields.test_items[0].item_id = fields.calibration_items[0].item_id.clone();
        assert!(validate_plan_fields(&fields).is_err());
    }

    #[test]
    fn rejects_missing_family_minima_and_zero_alpha() {
        let (plan, _, _) = make_plan();
        let mut fields = plan.fields();
        fields.familywise_alpha_millionths = 0;
        assert!(validate_plan_fields(&fields).is_err());
        fields.familywise_alpha_millionths = 50_000;
        fields.strata[0].minimum_calibration_failure_items = 20;
        assert!(validate_plan_fields(&fields).is_err());
    }

    #[test]
    fn judge_uninformative_when_lower_sensitivity_does_not_exceed_lower_false_positive_rate() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        for observation in value["calibration_observations"].as_array_mut().unwrap() {
            observation["judge_outcome"] = serde_json::json!("failure");
        }
        let report = analyze_imperfect_judge_certification(&plan, &serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(report.fields.family_results[0].status, FamilyCertificationStatusV1::JudgeUninformative);
        assert_eq!(report.fields.overall_status, OverallCertificationStatusV1::JudgeUninformative);
    }

    #[test]
    fn rejects_missing_duplicate_and_digest_tampered_observations() {
        let (_, plan, receipt) = make_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["test_observations"].as_array_mut().unwrap().pop();
        assert!(analyze_imperfect_judge_certification(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("missing planned test"));
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        let first = value["calibration_observations"][0].clone();
        value["calibration_observations"].as_array_mut().unwrap().push(first);
        assert!(analyze_imperfect_judge_certification(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("duplicate calibration"));
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["test_observations"][0]["input_sha256"] = serde_json::json!(digest("wrong"));
        assert!(analyze_imperfect_judge_certification(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("test input digest differs"));
    }

    #[test]
    fn channel_inversion_has_expected_conservative_edges() {
        assert_eq!(invert_judge_channel(100_000, 900_000, 100_000), Some(0));
        assert_eq!(invert_judge_channel(500_000, 900_000, 100_000), Some(500_000));
        assert_eq!(invert_judge_channel(900_000, 500_000, 500_000), None);
    }

    #[test]
    fn one_sided_hoeffding_radius_decreases_with_sample_size() {
        let small = hoeffding_radius_millionths(100, 0.01);
        let large = hoeffding_radius_millionths(10_000, 0.01);
        assert!(large < small);
    }
}
