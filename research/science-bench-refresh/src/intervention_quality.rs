//! Independent quality audit for generated evaluator interventions.
//!
//! This module evaluates the intervention-generation pipeline, not the judge
//! under test. It requires a census of generated candidates and a blinded
//! adjudication panel. Hashes bind declared artifacts; they do not prove
//! reviewer independence, raw-artifact custody, or semantic truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const QUALITY_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_intervention_quality_plan.v1";
pub const QUALITY_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_intervention_quality_receipt.v1";
pub const QUALITY_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_intervention_quality_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.intervention_quality.plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.intervention_quality.report.v1\0";
const MAX_CANDIDATES: usize = 100_000;
const MAX_PANEL: u32 = 7;
const MAX_LABEL_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QualityPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl QualityPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum IntendedSemanticRelationV1 {
    SemanticInvariant,
    SemanticChange,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PipelineDispositionV1 {
    Accepted,
    Rejected,
    Abstained,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ReviewerVerdictV1 {
    MeaningPreserving,
    MeaningChanging,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdjudicatedRelationV1 {
    Valid,
    Invalid,
    Ambiguous,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationPolicyV1 {
    pub reviewers_per_candidate: u32,
    /// A single verdict must receive at least this share of all panel votes.
    pub consensus_threshold_permille: u16,
    pub require_blind_to_generator_identity: bool,
    pub require_blind_to_intended_relation: bool,
    pub require_blind_to_pipeline_disposition: bool,
    pub require_blind_to_evaluator_outcome: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionCandidateV1 {
    pub candidate_id: String,
    pub transformation_family: String,
    pub target_failure_mode: String,
    pub semantic_cluster_sha256: String,
    pub intended_relation: IntendedSemanticRelationV1,
    pub pipeline_disposition: PipelineDispositionV1,
    pub baseline_input_sha256: String,
    pub variant_input_sha256: String,
    pub generation_tokens: u64,
    pub generation_cost_micro_usd: u64,
    pub generation_wall_clock_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionQualityPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: QualityPurposeV1,
    pub study_id: String,
    pub generator_source_revision: String,
    pub generator_prompt_sha256: String,
    pub semantic_verifier_source_revision: String,
    pub semantic_verifier_policy_sha256: String,
    pub candidate_sampling_policy: String,
    pub adjudication_policy: AdjudicationPolicyV1,
    pub candidates: Vec<InterventionCandidateV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionQualityPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: QualityPurposeV1,
    pub study_id: String,
    pub generator_source_revision: String,
    pub generator_prompt_sha256: String,
    pub semantic_verifier_source_revision: String,
    pub semantic_verifier_policy_sha256: String,
    pub candidate_sampling_policy: String,
    pub adjudication_policy: AdjudicationPolicyV1,
    pub candidates: Vec<InterventionCandidateV1>,
    pub plan_commitment_sha256: String,
}

impl InterventionQualityPlanV1 {
    fn fields(&self) -> InterventionQualityPlanFieldsV1 {
        InterventionQualityPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            study_id: self.study_id.clone(),
            generator_source_revision: self.generator_source_revision.clone(),
            generator_prompt_sha256: self.generator_prompt_sha256.clone(),
            semantic_verifier_source_revision: self.semantic_verifier_source_revision.clone(),
            semantic_verifier_policy_sha256: self.semantic_verifier_policy_sha256.clone(),
            candidate_sampling_policy: self.candidate_sampling_policy.clone(),
            adjudication_policy: self.adjudication_policy.clone(),
            candidates: self.candidates.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HumanAdjudicationV1 {
    pub reviewer_id_sha256: String,
    pub verdict: ReviewerVerdictV1,
    pub blind_to_generator_identity: bool,
    pub blind_to_intended_relation: bool,
    pub blind_to_pipeline_disposition: bool,
    pub blind_to_evaluator_outcome: bool,
    pub adjudication_artifact_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidateAdjudicationV1 {
    pub candidate_id: String,
    pub reviewers: Vec<HumanAdjudicationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionQualityReceiptV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: QualityPurposeV1,
    pub run_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub generator_source_revision: String,
    pub semantic_verifier_source_revision: String,
    pub semantic_verifier_policy_sha256: String,
    pub adjudications: Vec<CandidateAdjudicationV1>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FamilyQualityMetricsV1 {
    pub transformation_family: String,
    pub candidate_count: u64,
    pub accepted_count: u64,
    pub accepted_valid_count: u64,
    pub accepted_invalid_count: u64,
    pub accepted_ambiguous_count: u64,
    pub rejected_valid_count: u64,
    pub rejected_invalid_count: u64,
    pub rejected_ambiguous_count: u64,
    pub abstained_valid_count: u64,
    pub abstained_invalid_count: u64,
    pub abstained_ambiguous_count: u64,
    pub abstained_count: u64,
    pub unique_semantic_cluster_count: u64,
    pub unique_valid_accepted_test_unit_count: u64,
    pub generation_tokens: u64,
    pub generation_cost_micro_usd: u64,
    pub valid_unique_accepted_per_1000_tokens: Option<f64>,
    pub valid_unique_accepted_per_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FailureModeQualityMetricsV1 {
    pub target_failure_mode: String,
    pub candidate_count: u64,
    pub accepted_valid_count: u64,
    pub accepted_invalid_count: u64,
    pub accepted_ambiguous_count: u64,
    pub rejected_valid_count: u64,
    pub rejected_invalid_count: u64,
    pub rejected_ambiguous_count: u64,
    pub abstained_valid_count: u64,
    pub abstained_invalid_count: u64,
    pub abstained_ambiguous_count: u64,
    pub abstained_count: u64,
    pub unique_valid_accepted_test_unit_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct InterventionQualityReportFieldsV1 {
    pub schema: String,
    pub study_id: String,
    pub analyzer_source_revision: String,
    pub plan_sha256: String,
    pub receipt_sha256: String,
    pub plan_commitment_sha256: String,
    pub candidate_count: u64,
    pub adjudicated_candidate_count: u64,
    pub accepted_count: u64,
    pub rejected_count: u64,
    pub abstained_count: u64,
    pub valid_relation_count: u64,
    pub invalid_relation_count: u64,
    pub ambiguous_relation_count: u64,
    pub accepted_valid_count: u64,
    pub accepted_invalid_count: u64,
    pub accepted_ambiguous_count: u64,
    pub rejected_valid_count: u64,
    pub rejected_invalid_count: u64,
    pub rejected_ambiguous_count: u64,
    pub abstained_valid_count: u64,
    pub abstained_invalid_count: u64,
    pub abstained_ambiguous_count: u64,
    /// Lower bound: ambiguous accepted candidates remain in the denominator,
    /// and are not counted as validated successes.
    pub accepted_valid_precision_lower_bound_ppm: Option<u32>,
    pub accepted_confirmed_invalid_rate_ppm: Option<u32>,
    pub accepted_ambiguous_rate_ppm: Option<u32>,
    pub valid_candidate_recall_ppm: Option<u32>,
    pub pipeline_acceptance_rate_ppm: Option<u32>,
    pub reviewer_pairwise_agreement_ppm: Option<u32>,
    pub reviewer_pairwise_comparison_count: u64,
    pub reviewer_fleiss_kappa: Option<f64>,
    pub unique_transformation_family_count: u64,
    pub unique_failure_mode_count: u64,
    pub unique_semantic_cluster_count: u64,
    pub unique_valid_accepted_test_unit_count: u64,
    pub generation_tokens: u64,
    pub generation_cost_micro_usd: u64,
    pub generation_wall_clock_ms: u64,
    pub valid_unique_accepted_per_1000_tokens: Option<f64>,
    pub valid_unique_accepted_per_usd: Option<f64>,
    pub family_metrics: Vec<FamilyQualityMetricsV1>,
    pub failure_mode_metrics: Vec<FailureModeQualityMetricsV1>,
    pub reviewer_independence_proven: bool,
    pub raw_adjudication_custody_proven: bool,
    pub semantic_truth_proven: bool,
    pub execution_authenticity_proven: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct InterventionQualityReportV1 {
    #[serde(flatten)]
    pub fields: InterventionQualityReportFieldsV1,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Default, Clone)]
struct DispositionCountsV1 {
    accepted_valid: u64,
    accepted_invalid: u64,
    accepted_ambiguous: u64,
    rejected_valid: u64,
    rejected_invalid: u64,
    rejected_ambiguous: u64,
    abstained_valid: u64,
    abstained_invalid: u64,
    abstained_ambiguous: u64,
}

#[derive(Debug, Default, Clone)]
struct GroupAccumulatorV1 {
    candidate_count: u64,
    disposition: DispositionCountsV1,
    cluster_ids: BTreeSet<String>,
    valid_units: BTreeSet<(String, String)>,
    generation_tokens: u64,
    generation_cost_micro_usd: u64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_domain(domain: &[u8], payload: &[u8]) -> Result<String, String> {
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

fn nonempty(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_bytes
}

fn ratio_ppm(numerator: u64, denominator: u64) -> Option<u32> {
    if denominator == 0 {
        return None;
    }
    let value = (u128::from(numerator) * 1_000_000) / u128::from(denominator);
    u32::try_from(value.min(1_000_000)).ok()
}

fn per_1000_tokens(count: u64, tokens: u64) -> Option<f64> {
    (tokens > 0).then(|| count as f64 * 1_000.0 / tokens as f64)
}

fn per_usd(count: u64, cost_micro_usd: u64) -> Option<f64> {
    (cost_micro_usd > 0).then(|| count as f64 * 1_000_000.0 / cost_micro_usd as f64)
}

fn checked_add(total: &mut u64, value: u64, label: &str) -> Result<(), String> {
    *total = total.checked_add(value)
        .ok_or_else(|| format!("{label} total overflow"))?;
    Ok(())
}

fn plan_fields_commitment(fields: &InterventionQualityPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize intervention quality plan: {error}"))?;
    digest_domain(PLAN_DOMAIN, &payload)
}

/// Validate a census-based intervention-quality plan before commitments are accepted.
pub fn validate_intervention_quality_plan_fields(
    fields: &InterventionQualityPlanFieldsV1,
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();

    if fields.schema != QUALITY_PLAN_SCHEMA {
        issues.push("unsupported intervention quality plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push("analyzer_source_revision must match the exact compiled validator revision".to_owned());
    }
    if !nonempty(&fields.study_id, MAX_LABEL_BYTES) {
        issues.push("study_id must be non-empty and bounded".to_owned());
    }
    for (name, revision) in [
        ("generator_source_revision", &fields.generator_source_revision),
        ("semantic_verifier_source_revision", &fields.semantic_verifier_source_revision),
    ] {
        if !valid_revision(revision) && !(allow_zero && revision == &"0".repeat(40)) {
            issues.push(format!("{name} must be a full lowercase Git revision"));
        }
    }
    for (name, digest) in [
        ("generator_prompt_sha256", &fields.generator_prompt_sha256),
        ("semantic_verifier_policy_sha256", &fields.semantic_verifier_policy_sha256),
    ] {
        if !valid_digest(digest, allow_zero) {
            issues.push(format!("{name} must be a valid SHA-256 digest"));
        }
    }
    if fields.candidate_sampling_policy != "full_census" {
        issues.push("v1 requires full_census; sampled/adaptive adjudication needs a separately preregistered design".to_owned());
    }
    let policy = &fields.adjudication_policy;
    if !(3..=MAX_PANEL).contains(&policy.reviewers_per_candidate) {
        issues.push(format!("reviewers_per_candidate must be in 3..={MAX_PANEL}"));
    }
    if !(501..=1_000).contains(&policy.consensus_threshold_permille) {
        issues.push("consensus_threshold_permille must be in 501..=1000".to_owned());
    }
    if !policy.require_blind_to_generator_identity
        || !policy.require_blind_to_intended_relation
        || !policy.require_blind_to_pipeline_disposition
        || !policy.require_blind_to_evaluator_outcome
    {
        issues.push("v1 requires blinding to generator, pipeline disposition and evaluator outcome".to_owned());
    }
    if fields.candidates.is_empty() || fields.candidates.len() > MAX_CANDIDATES {
        issues.push(format!("candidates must contain 1..={MAX_CANDIDATES} records"));
    }

    let mut candidate_ids = BTreeSet::new();
    let mut cost_tokens = 0_u64;
    let mut cost_micro_usd = 0_u64;
    let mut wall_clock_ms = 0_u64;
    for candidate in &fields.candidates {
        if !nonempty(&candidate.candidate_id, MAX_LABEL_BYTES)
            || !candidate_ids.insert(candidate.candidate_id.as_str())
        {
            issues.push("candidate IDs must be non-empty, bounded and unique".to_owned());
        }
        for (name, value) in [
            ("transformation_family", &candidate.transformation_family),
            ("target_failure_mode", &candidate.target_failure_mode),
        ] {
            if !nonempty(value, MAX_LABEL_BYTES) {
                issues.push(format!("candidate {} {name} must be non-empty and bounded", candidate.candidate_id));
            }
        }
        for (name, digest) in [
            ("semantic_cluster_sha256", &candidate.semantic_cluster_sha256),
            ("baseline_input_sha256", &candidate.baseline_input_sha256),
            ("variant_input_sha256", &candidate.variant_input_sha256),
        ] {
            if !valid_digest(digest, allow_zero) {
                issues.push(format!("candidate {} has invalid {name}", candidate.candidate_id));
            }
        }
        if candidate.baseline_input_sha256 == candidate.variant_input_sha256 {
            issues.push(format!("candidate {} baseline and variant input digests are identical", candidate.candidate_id));
        }
        if let Err(error) = checked_add(&mut cost_tokens, candidate.generation_tokens, "generation token") {
            issues.push(error);
        }
        if let Err(error) = checked_add(&mut cost_micro_usd, candidate.generation_cost_micro_usd, "generation cost") {
            issues.push(error);
        }
        if let Err(error) = checked_add(&mut wall_clock_ms, candidate.generation_wall_clock_ms, "generation wall-clock") {
            issues.push(error);
        }
    }

    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

pub fn compute_intervention_quality_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: InterventionQualityPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("intervention quality plan fields invalid: {error}"))?;
    validate_intervention_quality_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    plan_fields_commitment(&fields)
}

fn verify_plan(plan_bytes: &[u8]) -> Result<(InterventionQualityPlanV1, String), String> {
    let plan: InterventionQualityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("intervention quality plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_intervention_quality_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    let expected = plan_fields_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("intervention quality plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

/// Verify and expose the immutable parent-plan binding needed by target-delta studies.
pub fn intervention_quality_plan_binding(
    plan_bytes: &[u8],
) -> Result<(String, String, String, String, String), String> {
    let (plan, plan_sha256) = verify_plan(plan_bytes)?;
    let purpose = if plan.purpose.qualified() {
        "qualified_run"
    } else {
        "example_only"
    };
    Ok((
        plan_sha256,
        plan.plan_commitment_sha256,
        plan.analyzer_source_revision,
        plan.study_id,
        purpose.to_owned(),
    ))
}

fn majority_relation(
    intended: IntendedSemanticRelationV1,
    reviewers: &[HumanAdjudicationV1],
    threshold_permille: u16,
) -> AdjudicatedRelationV1 {
    let mut preserving = 0_u64;
    let mut changing = 0_u64;
    let mut ambiguous = 0_u64;
    for reviewer in reviewers {
        match reviewer.verdict {
            ReviewerVerdictV1::MeaningPreserving => preserving += 1,
            ReviewerVerdictV1::MeaningChanging => changing += 1,
            ReviewerVerdictV1::Ambiguous => ambiguous += 1,
        }
    }
    let total = reviewers.len() as u64;
    let (winner, count) = [
        (ReviewerVerdictV1::MeaningPreserving, preserving),
        (ReviewerVerdictV1::MeaningChanging, changing),
        (ReviewerVerdictV1::Ambiguous, ambiguous),
    ]
    .into_iter()
    .max_by_key(|(_, count)| *count)
    .unwrap_or((ReviewerVerdictV1::Ambiguous, 0));
    if count * 1_000 < u64::from(threshold_permille) * total {
        return AdjudicatedRelationV1::Ambiguous;
    }
    match (intended, winner) {
        (_, ReviewerVerdictV1::Ambiguous) => AdjudicatedRelationV1::Ambiguous,
        (IntendedSemanticRelationV1::SemanticInvariant, ReviewerVerdictV1::MeaningPreserving)
        | (IntendedSemanticRelationV1::SemanticChange, ReviewerVerdictV1::MeaningChanging) => {
            AdjudicatedRelationV1::Valid
        }
        _ => AdjudicatedRelationV1::Invalid,
    }
}

fn pairwise_agreement(reviewers: &[HumanAdjudicationV1]) -> (u64, u64) {
    let mut matches = 0_u64;
    let mut comparisons = 0_u64;
    for left in 0..reviewers.len() {
        for right in (left + 1)..reviewers.len() {
            comparisons += 1;
            if reviewers[left].verdict == reviewers[right].verdict {
                matches += 1;
            }
        }
    }
    (matches, comparisons)
}

fn fleiss_kappa(vote_counts: &[[u64; 3]], reviewers_per_candidate: u32) -> Option<f64> {
    if vote_counts.is_empty() || reviewers_per_candidate < 2 {
        return None;
    }
    let panel_size = reviewers_per_candidate as f64;
    let pair_denominator = panel_size * (panel_size - 1.0);
    let mut category_totals = [0_u64; 3];
    let mut mean_observed_agreement = 0.0;
    for counts in vote_counts {
        let assigned = counts.iter().sum::<u64>();
        if assigned != u64::from(reviewers_per_candidate) {
            return None;
        }
        for (total, count) in category_totals.iter_mut().zip(counts) {
            *total += *count;
        }
        let squared_sum = counts.iter().map(|count| (*count as f64).powi(2)).sum::<f64>();
        mean_observed_agreement += (squared_sum - panel_size) / pair_denominator;
    }
    mean_observed_agreement /= vote_counts.len() as f64;
    let total_votes = (vote_counts.len() as u64) * u64::from(reviewers_per_candidate);
    let expected_agreement = category_totals.iter()
        .map(|count| {
            let proportion = *count as f64 / total_votes as f64;
            proportion * proportion
        })
        .sum::<f64>();
    (1.0 - expected_agreement > f64::EPSILON)
        .then(|| (mean_observed_agreement - expected_agreement) / (1.0 - expected_agreement))
}

fn add_disposition(
    counts: &mut DispositionCountsV1,
    disposition: PipelineDispositionV1,
    relation: AdjudicatedRelationV1,
) {
    match (disposition, relation) {
        (PipelineDispositionV1::Accepted, AdjudicatedRelationV1::Valid) => counts.accepted_valid += 1,
        (PipelineDispositionV1::Accepted, AdjudicatedRelationV1::Invalid) => counts.accepted_invalid += 1,
        (PipelineDispositionV1::Accepted, AdjudicatedRelationV1::Ambiguous) => counts.accepted_ambiguous += 1,
        (PipelineDispositionV1::Rejected, AdjudicatedRelationV1::Valid) => counts.rejected_valid += 1,
        (PipelineDispositionV1::Rejected, AdjudicatedRelationV1::Invalid) => counts.rejected_invalid += 1,
        (PipelineDispositionV1::Rejected, AdjudicatedRelationV1::Ambiguous) => counts.rejected_ambiguous += 1,
        (PipelineDispositionV1::Abstained, AdjudicatedRelationV1::Valid) => counts.abstained_valid += 1,
        (PipelineDispositionV1::Abstained, AdjudicatedRelationV1::Invalid) => counts.abstained_invalid += 1,
        (PipelineDispositionV1::Abstained, AdjudicatedRelationV1::Ambiguous) => counts.abstained_ambiguous += 1,
    }
}

fn family_report(
    family: String,
    aggregate: GroupAccumulatorV1,
) -> FamilyQualityMetricsV1 {
    let valid_unique = aggregate.valid_units.len() as u64;
    FamilyQualityMetricsV1 {
        transformation_family: family,
        candidate_count: aggregate.candidate_count,
        accepted_count: aggregate.disposition.accepted_valid
            + aggregate.disposition.accepted_invalid
            + aggregate.disposition.accepted_ambiguous,
        accepted_valid_count: aggregate.disposition.accepted_valid,
        accepted_invalid_count: aggregate.disposition.accepted_invalid,
        accepted_ambiguous_count: aggregate.disposition.accepted_ambiguous,
        rejected_valid_count: aggregate.disposition.rejected_valid,
        rejected_invalid_count: aggregate.disposition.rejected_invalid,
        rejected_ambiguous_count: aggregate.disposition.rejected_ambiguous,
        abstained_valid_count: aggregate.disposition.abstained_valid,
        abstained_invalid_count: aggregate.disposition.abstained_invalid,
        abstained_ambiguous_count: aggregate.disposition.abstained_ambiguous,
        abstained_count: aggregate.disposition.abstained_valid
            + aggregate.disposition.abstained_invalid
            + aggregate.disposition.abstained_ambiguous,
        unique_semantic_cluster_count: aggregate.cluster_ids.len() as u64,
        unique_valid_accepted_test_unit_count: valid_unique,
        generation_tokens: aggregate.generation_tokens,
        generation_cost_micro_usd: aggregate.generation_cost_micro_usd,
        valid_unique_accepted_per_1000_tokens: per_1000_tokens(valid_unique, aggregate.generation_tokens),
        valid_unique_accepted_per_usd: per_usd(valid_unique, aggregate.generation_cost_micro_usd),
    }
}

/// Audit the full candidate census against a blinded human adjudication receipt.
pub fn analyze_intervention_quality(
    plan_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<InterventionQualityReportV1, String> {
    let (plan, plan_sha256) = verify_plan(plan_bytes)?;
    let receipt: InterventionQualityReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("intervention quality receipt invalid: {error}"))?;
    if receipt.schema != QUALITY_RECEIPT_SCHEMA {
        return Err("unsupported intervention quality receipt schema".to_owned());
    }
    if receipt.purpose != plan.purpose {
        return Err("receipt purpose differs from plan".to_owned());
    }
    if receipt.analyzer_source_revision != plan.analyzer_source_revision
        || receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        return Err("receipt analyzer revision differs from the exact compiled validator revision".to_owned());
    }
    if !nonempty(&receipt.run_id, MAX_LABEL_BYTES) {
        return Err("receipt run_id must be non-empty and bounded".to_owned());
    }
    if receipt.plan_sha256 != plan_sha256 || receipt.plan_commitment_sha256 != plan.plan_commitment_sha256 {
        return Err("receipt does not bind exact plan bytes and commitment".to_owned());
    }
    if receipt.generator_source_revision != plan.generator_source_revision
        || receipt.semantic_verifier_source_revision != plan.semantic_verifier_source_revision
        || receipt.semantic_verifier_policy_sha256 != plan.semantic_verifier_policy_sha256
    {
        return Err("receipt generator/verifier revision or policy differs from frozen plan".to_owned());
    }
    if receipt.adjudications.len() != plan.candidates.len() {
        return Err("receipt must contain exactly one adjudication for every generated candidate".to_owned());
    }

    let mut adjudications = BTreeMap::new();
    for adjudication in &receipt.adjudications {
        if adjudications.insert(adjudication.candidate_id.as_str(), adjudication).is_some() {
            return Err(format!("duplicate candidate adjudication {}", adjudication.candidate_id));
        }
    }

    let mut family_aggregates: BTreeMap<String, GroupAccumulatorV1> = BTreeMap::new();
    let mut failure_aggregates: BTreeMap<String, GroupAccumulatorV1> = BTreeMap::new();
    let mut all_clusters = BTreeSet::new();
    let mut all_families = BTreeSet::new();
    let mut all_failure_modes = BTreeSet::new();
    let mut valid_accepted_units = BTreeSet::new();
    let mut total = DispositionCountsV1::default();
    let mut valid_relation_count = 0_u64;
    let mut invalid_relation_count = 0_u64;
    let mut ambiguous_relation_count = 0_u64;
    let mut reviewer_agreement_matches = 0_u64;
    let mut reviewer_pairwise_comparisons = 0_u64;
    let mut reviewer_vote_counts = Vec::new();
    let mut total_tokens = 0_u64;
    let mut total_cost = 0_u64;
    let mut total_wall_clock = 0_u64;
    let candidates_by_id: BTreeMap<&str, &InterventionCandidateV1> = plan.candidates.iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect();

    for adjudication in &receipt.adjudications {
        let candidate = candidates_by_id.get(adjudication.candidate_id.as_str())
            .ok_or_else(|| format!("unknown candidate adjudication {}", adjudication.candidate_id))?;
        let policy = &plan.adjudication_policy;
        if adjudication.reviewers.len() != policy.reviewers_per_candidate as usize
            || adjudication.reviewers.len() > MAX_PANEL as usize
        {
            return Err(format!("candidate {} must have exactly the preregistered reviewer count", candidate.candidate_id));
        }
        let mut reviewer_ids = BTreeSet::new();
        for reviewer in &adjudication.reviewers {
            if !valid_digest(&reviewer.reviewer_id_sha256, false)
                || !reviewer_ids.insert(reviewer.reviewer_id_sha256.as_str())
            {
                return Err(format!("candidate {} has malformed or duplicate reviewer IDs", candidate.candidate_id));
            }
            if !valid_digest(&reviewer.adjudication_artifact_sha256, false) {
                return Err(format!("candidate {} has invalid adjudication artifact digest", candidate.candidate_id));
            }
            if (policy.require_blind_to_generator_identity && !reviewer.blind_to_generator_identity)
                || (policy.require_blind_to_intended_relation && !reviewer.blind_to_intended_relation)
                || (policy.require_blind_to_pipeline_disposition && !reviewer.blind_to_pipeline_disposition)
                || (policy.require_blind_to_evaluator_outcome && !reviewer.blind_to_evaluator_outcome)
            {
                return Err(format!("candidate {} contains a non-blinded adjudication", candidate.candidate_id));
            }
        }
        let relation = majority_relation(
            candidate.intended_relation,
            &adjudication.reviewers,
            policy.consensus_threshold_permille,
        );
        let mut candidate_vote_counts = [0_u64; 3];
        for reviewer in &adjudication.reviewers {
            match reviewer.verdict {
                ReviewerVerdictV1::MeaningPreserving => candidate_vote_counts[0] += 1,
                ReviewerVerdictV1::MeaningChanging => candidate_vote_counts[1] += 1,
                ReviewerVerdictV1::Ambiguous => candidate_vote_counts[2] += 1,
            }
        }
        reviewer_vote_counts.push(candidate_vote_counts);
        let (matches, comparisons) = pairwise_agreement(&adjudication.reviewers);
        checked_add(&mut reviewer_agreement_matches, matches, "reviewer agreement")?;
        checked_add(&mut reviewer_pairwise_comparisons, comparisons, "reviewer comparison")?;
        match relation {
            AdjudicatedRelationV1::Valid => valid_relation_count += 1,
            AdjudicatedRelationV1::Invalid => invalid_relation_count += 1,
            AdjudicatedRelationV1::Ambiguous => ambiguous_relation_count += 1,
        }

        add_disposition(&mut total, candidate.pipeline_disposition, relation);
        let family = family_aggregates.entry(candidate.transformation_family.clone()).or_default();
        let failure = failure_aggregates.entry(candidate.target_failure_mode.clone()).or_default();
        for aggregate in [family, failure] {
            aggregate.candidate_count += 1;
            aggregate.cluster_ids.insert(candidate.semantic_cluster_sha256.clone());
            checked_add(&mut aggregate.generation_tokens, candidate.generation_tokens, "group token")?;
            checked_add(&mut aggregate.generation_cost_micro_usd, candidate.generation_cost_micro_usd, "group cost")?;
            add_disposition(&mut aggregate.disposition, candidate.pipeline_disposition, relation);
            if candidate.pipeline_disposition == PipelineDispositionV1::Accepted
                && relation == AdjudicatedRelationV1::Valid
            {
                aggregate.valid_units.insert((
                    candidate.semantic_cluster_sha256.clone(),
                    candidate.target_failure_mode.clone(),
                ));
            }
        }
        if candidate.pipeline_disposition == PipelineDispositionV1::Accepted
            && relation == AdjudicatedRelationV1::Valid
        {
            valid_accepted_units.insert((
                candidate.semantic_cluster_sha256.clone(),
                candidate.transformation_family.clone(),
                candidate.target_failure_mode.clone(),
            ));
        }
        all_clusters.insert(candidate.semantic_cluster_sha256.clone());
        all_families.insert(candidate.transformation_family.clone());
        all_failure_modes.insert(candidate.target_failure_mode.clone());
        checked_add(&mut total_tokens, candidate.generation_tokens, "generation token")?;
        checked_add(&mut total_cost, candidate.generation_cost_micro_usd, "generation cost")?;
        checked_add(&mut total_wall_clock, candidate.generation_wall_clock_ms, "generation wall-clock")?;
    }

    if adjudications.len() != candidates_by_id.len()
        || candidates_by_id.keys().any(|candidate_id| !adjudications.contains_key(candidate_id))
    {
        return Err("receipt is missing one or more generated candidate adjudications".to_owned());
    }

    let accepted = total.accepted_valid + total.accepted_invalid + total.accepted_ambiguous;
    let rejected = total.rejected_valid + total.rejected_invalid + total.rejected_ambiguous;
    let abstained = total.abstained_valid + total.abstained_invalid + total.abstained_ambiguous;
    let valid_count = valid_relation_count;
    let valid_accepted = total.accepted_valid;
    let unique_valid_accepted = valid_accepted_units.len() as u64;
    let family_metrics = family_aggregates.into_iter()
        .map(|(name, aggregate)| family_report(name, aggregate))
        .collect();
    let failure_mode_metrics = failure_aggregates.into_iter()
        .map(|(name, aggregate)| {
            let valid_units = aggregate.valid_units.len() as u64;
            FailureModeQualityMetricsV1 {
                target_failure_mode: name,
                candidate_count: aggregate.candidate_count,
                accepted_valid_count: aggregate.disposition.accepted_valid,
                accepted_invalid_count: aggregate.disposition.accepted_invalid,
                accepted_ambiguous_count: aggregate.disposition.accepted_ambiguous,
                rejected_valid_count: aggregate.disposition.rejected_valid,
                rejected_invalid_count: aggregate.disposition.rejected_invalid,
                rejected_ambiguous_count: aggregate.disposition.rejected_ambiguous,
                abstained_valid_count: aggregate.disposition.abstained_valid,
                abstained_invalid_count: aggregate.disposition.abstained_invalid,
                abstained_ambiguous_count: aggregate.disposition.abstained_ambiguous,
                abstained_count: aggregate.disposition.abstained_valid
                    + aggregate.disposition.abstained_invalid
                    + aggregate.disposition.abstained_ambiguous,
                unique_valid_accepted_test_unit_count: valid_units,
            }
        })
        .collect();

    let fields = InterventionQualityReportFieldsV1 {
        schema: QUALITY_REPORT_SCHEMA.to_owned(),
        study_id: plan.study_id,
        analyzer_source_revision: plan.analyzer_source_revision,
        plan_sha256,
        receipt_sha256: sha256_hex(receipt_bytes),
        plan_commitment_sha256: plan.plan_commitment_sha256,
        candidate_count: plan.candidates.len() as u64,
        adjudicated_candidate_count: adjudications.len() as u64,
        accepted_count: accepted,
        rejected_count: rejected,
        abstained_count: abstained,
        valid_relation_count,
        invalid_relation_count,
        ambiguous_relation_count,
        accepted_valid_count: total.accepted_valid,
        accepted_invalid_count: total.accepted_invalid,
        accepted_ambiguous_count: total.accepted_ambiguous,
        rejected_valid_count: total.rejected_valid,
        rejected_invalid_count: total.rejected_invalid,
        rejected_ambiguous_count: total.rejected_ambiguous,
        abstained_valid_count: total.abstained_valid,
        abstained_invalid_count: total.abstained_invalid,
        abstained_ambiguous_count: total.abstained_ambiguous,
        accepted_valid_precision_lower_bound_ppm: ratio_ppm(total.accepted_valid, accepted),
        accepted_confirmed_invalid_rate_ppm: ratio_ppm(total.accepted_invalid, accepted),
        accepted_ambiguous_rate_ppm: ratio_ppm(total.accepted_ambiguous, accepted),
        valid_candidate_recall_ppm: ratio_ppm(valid_accepted, valid_count),
        pipeline_acceptance_rate_ppm: ratio_ppm(accepted, plan.candidates.len() as u64),
        reviewer_pairwise_agreement_ppm: ratio_ppm(reviewer_agreement_matches, reviewer_pairwise_comparisons),
        reviewer_pairwise_comparison_count: reviewer_pairwise_comparisons,
        reviewer_fleiss_kappa: fleiss_kappa(
            &reviewer_vote_counts,
            plan.adjudication_policy.reviewers_per_candidate,
        ),
        unique_transformation_family_count: all_families.len() as u64,
        unique_failure_mode_count: all_failure_modes.len() as u64,
        unique_semantic_cluster_count: all_clusters.len() as u64,
        unique_valid_accepted_test_unit_count: unique_valid_accepted,
        generation_tokens: total_tokens,
        generation_cost_micro_usd: total_cost,
        generation_wall_clock_ms: total_wall_clock,
        valid_unique_accepted_per_1000_tokens: per_1000_tokens(unique_valid_accepted, total_tokens),
        valid_unique_accepted_per_usd: per_usd(unique_valid_accepted, total_cost),
        family_metrics,
        failure_mode_metrics,
        reviewer_independence_proven: false,
        raw_adjudication_custody_proven: false,
        semantic_truth_proven: false,
        execution_authenticity_proven: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        claim_boundary: "This report audits a full declared candidate census using supplied blinded-panel votes. It does not prove reviewer independence, raw artifact custody, semantic truth, randomization, generator execution authenticity, downstream judge behavior, causal effects, generalization, or scoring eligibility. Ambiguous panel outcomes are never counted as validated successes.".to_owned(),
    };
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize intervention quality report: {error}"))?;
    let commitment = digest_domain(REPORT_DOMAIN, &payload)?;
    Ok(InterventionQualityReportV1 {
        fields,
        report_commitment_sha256: commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn fields() -> InterventionQualityPlanFieldsV1 {
        let make_candidate = |id: &str, family: &str, failure: &str, cluster: &str,
                              relation: IntendedSemanticRelationV1, disposition: PipelineDispositionV1,
                              cost: u64| InterventionCandidateV1 {
            candidate_id: id.to_owned(),
            transformation_family: family.to_owned(),
            target_failure_mode: failure.to_owned(),
            semantic_cluster_sha256: digest(cluster),
            intended_relation: relation,
            pipeline_disposition: disposition,
            baseline_input_sha256: digest(&format!("{id}-base")),
            variant_input_sha256: digest(&format!("{id}-variant")),
            generation_tokens: cost,
            generation_cost_micro_usd: cost * 10,
            generation_wall_clock_ms: cost * 2,
        };
        InterventionQualityPlanFieldsV1 {
            schema: QUALITY_PLAN_SCHEMA.to_owned(),
            analyzer_source_revision: crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            purpose: QualityPurposeV1::QualifiedRun,
            study_id: "quality-study-001".to_owned(),
            generator_source_revision: "a".repeat(40),
            generator_prompt_sha256: digest("generator-prompt"),
            semantic_verifier_source_revision: "b".repeat(40),
            semantic_verifier_policy_sha256: digest("semantic-policy"),
            candidate_sampling_policy: "full_census".to_owned(),
            adjudication_policy: AdjudicationPolicyV1 {
                reviewers_per_candidate: 3,
                consensus_threshold_permille: 666,
                require_blind_to_generator_identity: true,
                require_blind_to_intended_relation: true,
                require_blind_to_pipeline_disposition: true,
                require_blind_to_evaluator_outcome: true,
            },
            candidates: vec![
                make_candidate("c1", "surface", "position", "task-1", IntendedSemanticRelationV1::SemanticInvariant, PipelineDispositionV1::Accepted, 100),
                make_candidate("c2", "surface", "format", "task-1", IntendedSemanticRelationV1::SemanticChange, PipelineDispositionV1::Accepted, 120),
                make_candidate("c3", "evidence", "withdrawal", "task-2", IntendedSemanticRelationV1::SemanticChange, PipelineDispositionV1::Rejected, 80),
                make_candidate("c4", "authority", "provenance", "task-3", IntendedSemanticRelationV1::SemanticInvariant, PipelineDispositionV1::Abstained, 60),
                make_candidate("c5", "authority", "authority-cue", "task-4", IntendedSemanticRelationV1::SemanticChange, PipelineDispositionV1::Accepted, 70),
                make_candidate("c6", "surface", "rubric-order", "task-5", IntendedSemanticRelationV1::SemanticInvariant, PipelineDispositionV1::Accepted, 90),
            ],
        }
    }

    fn reviewers(candidate: &str, votes: [ReviewerVerdictV1; 3]) -> Vec<HumanAdjudicationV1> {
        votes.into_iter().enumerate().map(|(index, verdict)| HumanAdjudicationV1 {
            reviewer_id_sha256: digest(&format!("reviewer-{candidate}-{index}")),
            verdict,
            blind_to_generator_identity: true,
            blind_to_intended_relation: true,
            blind_to_pipeline_disposition: true,
            blind_to_evaluator_outcome: true,
            adjudication_artifact_sha256: digest(&format!("artifact-{candidate}-{index}")),
        }).collect()
    }

    fn fixture() -> (InterventionQualityPlanV1, Vec<u8>, Vec<u8>) {
        let fields = fields();
        validate_intervention_quality_plan_fields(&fields).unwrap();
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = InterventionQualityPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            study_id: fields.study_id.clone(),
            generator_source_revision: fields.generator_source_revision.clone(),
            generator_prompt_sha256: fields.generator_prompt_sha256.clone(),
            semantic_verifier_source_revision: fields.semantic_verifier_source_revision.clone(),
            semantic_verifier_policy_sha256: fields.semantic_verifier_policy_sha256.clone(),
            candidate_sampling_policy: fields.candidate_sampling_policy.clone(),
            adjudication_policy: fields.adjudication_policy.clone(),
            candidates: fields.candidates.clone(),
            plan_commitment_sha256: commitment,
        };
        let plan_bytes = serde_json::to_vec_pretty(&plan).unwrap();
        let receipt = InterventionQualityReceiptV1 {
            schema: QUALITY_RECEIPT_SCHEMA.to_owned(),
            analyzer_source_revision: plan.analyzer_source_revision.clone(),
            purpose: plan.purpose,
            run_id: "synthetic-run-001".to_owned(),
            plan_sha256: sha256_hex(&plan_bytes),
            plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
            generator_source_revision: plan.generator_source_revision.clone(),
            semantic_verifier_source_revision: plan.semantic_verifier_source_revision.clone(),
            semantic_verifier_policy_sha256: plan.semantic_verifier_policy_sha256.clone(),
            adjudications: vec![
                CandidateAdjudicationV1 {
                    candidate_id: "c1".to_owned(),
                    reviewers: reviewers("c1", [ReviewerVerdictV1::MeaningPreserving; 3]),
                },
                CandidateAdjudicationV1 {
                    candidate_id: "c2".to_owned(),
                    reviewers: reviewers("c2", [ReviewerVerdictV1::MeaningChanging, ReviewerVerdictV1::MeaningChanging, ReviewerVerdictV1::MeaningPreserving]),
                },
                CandidateAdjudicationV1 {
                    candidate_id: "c3".to_owned(),
                    reviewers: reviewers("c3", [ReviewerVerdictV1::MeaningChanging; 3]),
                },
                CandidateAdjudicationV1 {
                    candidate_id: "c4".to_owned(),
                    reviewers: reviewers("c4", [ReviewerVerdictV1::MeaningPreserving, ReviewerVerdictV1::MeaningChanging, ReviewerVerdictV1::Ambiguous]),
                },
                CandidateAdjudicationV1 {
                    candidate_id: "c5".to_owned(),
                    reviewers: reviewers("c5", [ReviewerVerdictV1::MeaningPreserving, ReviewerVerdictV1::MeaningChanging, ReviewerVerdictV1::Ambiguous]),
                },
                CandidateAdjudicationV1 {
                    candidate_id: "c6".to_owned(),
                    reviewers: reviewers("c6", [ReviewerVerdictV1::MeaningChanging, ReviewerVerdictV1::MeaningChanging, ReviewerVerdictV1::MeaningPreserving]),
                },
            ],
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        (plan, plan_bytes, receipt_bytes)
    }

    #[test]
    fn accepts_clean_census_and_counts_all_candidates() {
        let (_, plan, receipt) = fixture();
        let report = analyze_intervention_quality(&plan, &receipt).unwrap();
        assert_eq!(report.fields.candidate_count, 6);
        assert_eq!(report.fields.adjudicated_candidate_count, 6);
        assert_eq!(report.fields.accepted_valid_count, 2);
        assert_eq!(report.fields.rejected_valid_count, 1);
        assert_eq!(report.fields.abstained_count, 1);
        assert_eq!(report.fields.abstained_valid_count, 0);
        assert_eq!(report.fields.abstained_invalid_count, 0);
        assert_eq!(report.fields.abstained_ambiguous_count, 1);
        assert_eq!(report.fields.ambiguous_relation_count, 2);
        assert_eq!(report.fields.unique_valid_accepted_test_unit_count, 2);
        assert_eq!(report.fields.scoring_eligible_declared, false);
    }

    #[test]
    fn ambiguity_is_never_promoted_to_valid() {
        let (_, plan, receipt) = fixture();
        let report = analyze_intervention_quality(&plan, &receipt).unwrap();
        assert_eq!(report.fields.accepted_ambiguous_count, 1);
        assert_eq!(report.fields.accepted_invalid_count, 1);
        assert_eq!(report.fields.ambiguous_relation_count, 2);
    }

    #[test]
    fn reports_precision_recall_and_cost_denominators() {
        let (_, plan, receipt) = fixture();
        let report = analyze_intervention_quality(&plan, &receipt).unwrap();
        assert_eq!(report.fields.accepted_valid_precision_lower_bound_ppm, Some(500_000));
        assert_eq!(report.fields.valid_candidate_recall_ppm, Some(666_666));
        assert_eq!(report.fields.pipeline_acceptance_rate_ppm, Some(666_666));
        assert!(report.fields.valid_unique_accepted_per_1000_tokens.unwrap() > 0.0);
        assert!(report.fields.valid_unique_accepted_per_usd.unwrap() > 0.0);
        assert!(report.fields.reviewer_pairwise_agreement_ppm.unwrap() < 1_000_000);
        assert!(report.fields.reviewer_fleiss_kappa.is_some());
    }

    #[test]
    fn accepted_ambiguous_candidates_are_not_counted_as_valid_successes() {
        let (_, plan, receipt) = fixture();
        let report = analyze_intervention_quality(&plan, &receipt).unwrap();
        assert_eq!(report.fields.accepted_ambiguous_count, 1);
        assert_eq!(report.fields.accepted_valid_precision_lower_bound_ppm, Some(500_000));
        assert_eq!(report.fields.accepted_invalid_count, 1);
        assert_eq!(report.fields.accepted_ambiguous_rate_ppm, Some(250_000));
    }

    #[test]
    fn rejects_wrong_compiled_analyzer_revision() {
        let mut value = fields();
        value.analyzer_source_revision = "c".repeat(40);
        let issues = validate_intervention_quality_plan_fields(&value).unwrap_err();
        assert!(issues.iter().any(|issue| issue.contains("exact compiled validator revision")));
    }

    #[test]
    fn rejects_sampled_or_adaptive_candidates_in_v1() {
        let mut value = fields();
        value.candidate_sampling_policy = "adaptive_sample".to_owned();
        assert!(validate_intervention_quality_plan_fields(&value).is_err());
    }

    #[test]
    fn rejects_duplicate_candidates_and_identical_input_digests() {
        let mut value = fields();
        value.candidates.push(value.candidates[0].clone());
        assert!(validate_intervention_quality_plan_fields(&value).is_err());
        let mut value = fields();
        value.candidates[0].variant_input_sha256 = value.candidates[0].baseline_input_sha256.clone();
        assert!(validate_intervention_quality_plan_fields(&value).is_err());
    }

    #[test]
    fn rejects_missing_duplicate_unknown_and_nonblind_adjudications() {
        let (_, plan, receipt) = fixture();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["adjudications"].as_array_mut().unwrap().pop();
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("exactly one adjudication"));

        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        let first = value["adjudications"][0].clone();
        value["adjudications"].as_array_mut().unwrap().push(first);
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("duplicate"));

        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["adjudications"][0]["reviewers"][0]["blind_to_pipeline_disposition"] = serde_json::json!(false);
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("non-blinded"));
    }

    #[test]
    fn rejects_reviewers_exposed_to_the_intended_relation() {
        let (_, plan, receipt) = fixture();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["adjudications"][0]["reviewers"][0]["blind_to_intended_relation"] = serde_json::json!(false);
        let altered = serde_json::to_vec(&value).unwrap();
        assert!(analyze_intervention_quality(&plan, &altered).unwrap_err().contains("non-blinded"));
    }

    #[test]
    fn rejects_duplicate_reviewer_ids_and_bad_digests() {
        let (_, plan, receipt) = fixture();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        let id = value["adjudications"][0]["reviewers"][0]["reviewer_id_sha256"].clone();
        value["adjudications"][0]["reviewers"][1]["reviewer_id_sha256"] = id;
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("duplicate reviewer"));

        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["adjudications"][0]["reviewers"][0]["adjudication_artifact_sha256"] = serde_json::json!("z".repeat(64));
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("invalid adjudication"));
    }

    #[test]
    fn rejects_wrong_plan_digest_and_analyzer_revision() {
        let (_, plan, receipt) = fixture();
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["plan_sha256"] = serde_json::json!(digest("wrong plan"));
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("exact plan"));

        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["analyzer_source_revision"] = serde_json::json!("c".repeat(40));
        assert!(analyze_intervention_quality(&plan, &serde_json::to_vec(&value).unwrap()).unwrap_err().contains("exact compiled validator revision"));
    }

    #[test]
    fn report_is_deterministic_for_identical_plan_and_receipt_bytes() {
        let (_, plan, receipt) = fixture();
        let first = serde_json::to_vec(&analyze_intervention_quality(&plan, &receipt).unwrap()).unwrap();
        let second = serde_json::to_vec(&analyze_intervention_quality(&plan, &receipt).unwrap()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn cost_totals_overflow_are_rejected() {
        let mut value = fields();
        value.candidates[0].generation_tokens = u64::MAX;
        value.candidates[1].generation_tokens = 1;
        assert!(validate_intervention_quality_plan_fields(&value).is_err());
    }
}
