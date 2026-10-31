//! Independent semantic qualification and verified failure-yield accounting.
//!
//! This module audits supplied mutation plans, independent semantic-review
//! votes and evaluator outcomes. It does not authenticate reviewers, prove
//! reviewer blinding, custody raw artifact bytes, or establish population
//! validity. Evaluator failures count as verified only when the mutation is
//! unanimously confirmed against its preregistered semantic relation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MUTATION_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_mutation_validity_plan.v1";
pub const MUTATION_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_mutation_validity_receipt.v1";
pub const MUTATION_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_mutation_validity_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.mutation_validity.plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.mutation_validity.report.v1\0";
const MAX_VARIANTS: usize = 100_000;
const MAX_REVIEWERS: usize = 7;
const MAX_ARTIFACTS: usize = 32;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl MutationPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationSemanticRelationV1 {
    Preserving,
    Changing,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationFamilyV1 {
    Paraphrase,
    Formatting,
    EntitySubstitution,
    UnitChange,
    DateChange,
    ScopeChange,
    EvidenceAddition,
    EvidenceWithdrawal,
    SourceRetraction,
    MultimodalComposition,
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationModalityV1 {
    Text,
    Image,
    Audio,
    Video,
    Structured,
    Multimodal,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationArtifactCommitmentV1 {
    pub artifact_id: String,
    pub modality: MutationModalityV1,
    pub content_sha256: String,
    pub byte_length: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationVariantPlanV1 {
    pub mutation_id: String,
    pub metamorphic_pair_id: String,
    pub family: MutationFamilyV1,
    pub intended_relation: MutationSemanticRelationV1,
    pub target_dimension: Option<String>,
    pub baseline_input_sha256: String,
    pub variant_input_sha256: String,
    pub baseline_artifacts: Vec<MutationArtifactCommitmentV1>,
    pub variant_artifacts: Vec<MutationArtifactCommitmentV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: MutationPurposeV1,
    pub study_id: String,
    pub generator_source_revision: String,
    pub evaluator_source_revision: String,
    pub generator_policy_sha256: String,
    pub semantic_task_set_sha256: String,
    pub review_protocol_sha256: String,
    pub metamorphic_plan_sha256: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub reviewers_per_mutation: u32,
    pub variants: Vec<MutationVariantPlanV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: MutationPurposeV1,
    pub study_id: String,
    pub generator_source_revision: String,
    pub evaluator_source_revision: String,
    pub generator_policy_sha256: String,
    pub semantic_task_set_sha256: String,
    pub review_protocol_sha256: String,
    pub metamorphic_plan_sha256: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub reviewers_per_mutation: u32,
    pub variants: Vec<MutationVariantPlanV1>,
    pub plan_commitment_sha256: String,
}

impl MutationValidityPlanV1 {
    fn fields(&self) -> MutationValidityPlanFieldsV1 {
        MutationValidityPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            study_id: self.study_id.clone(),
            generator_source_revision: self.generator_source_revision.clone(),
            evaluator_source_revision: self.evaluator_source_revision.clone(),
            generator_policy_sha256: self.generator_policy_sha256.clone(),
            semantic_task_set_sha256: self.semantic_task_set_sha256.clone(),
            review_protocol_sha256: self.review_protocol_sha256.clone(),
            metamorphic_plan_sha256: self.metamorphic_plan_sha256.clone(),
            evaluator_prompt_sha256: self.evaluator_prompt_sha256.clone(),
            rubric_sha256: self.rubric_sha256.clone(),
            reviewers_per_mutation: self.reviewers_per_mutation,
            variants: self.variants.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticReviewDecisionV1 {
    PreservesSemantics,
    ChangesSemantics,
    Indeterminate,
    Abstain,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SemanticAdjudicationV1 {
    pub adjudicator_id_sha256: String,
    pub independence_group_sha256: String,
    pub decision: SemanticReviewDecisionV1,
    pub confidence_millionths: u32,
    pub rationale_sha256: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MutationEvaluationStatusV1 {
    Completed,
    TimedOut,
    Blocked,
    InvalidReceipt,
    NotStarted,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityObservationV1 {
    pub mutation_id: String,
    pub baseline_input_sha256: String,
    pub variant_input_sha256: String,
    pub adjudications: Vec<SemanticAdjudicationV1>,
    pub evaluation_status: MutationEvaluationStatusV1,
    pub evaluator_relation_pass: Option<bool>,
    pub failure_fingerprint_sha256: Option<String>,
    pub generator_call_count: u64,
    pub generator_token_count: u64,
    pub semantic_assistance_call_count: u64,
    pub semantic_assistance_token_count: u64,
    pub evaluator_call_count: u64,
    pub evaluator_token_count: u64,
    pub human_review_seconds: u64,
    pub wall_time_milliseconds: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MutationValidityReceiptV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: MutationPurposeV1,
    pub run_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub metamorphic_receipt_sha256: String,
    pub generator_source_revision: String,
    pub evaluator_source_revision: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub observations: Vec<MutationValidityObservationV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DerivedReviewStatus {
    ConfirmedValidPreserving,
    ConfirmedValidChanging,
    ConfirmedInvalid,
    Disputed,
    Insufficient,
    Abstained,
    Unreviewed,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MutationValiditySummaryV1 {
    pub mutation_id: String,
    pub intended_relation: MutationSemanticRelationV1,
    pub review_count: u64,
    pub pairwise_agreement_count: u64,
    pub pairwise_disagreement_count: u64,
    pub review_status: String,
    pub evaluation_status: MutationEvaluationStatusV1,
    pub evaluator_relation_pass: Option<bool>,
    pub failure_counted_as_verified: bool,
    pub failure_exclusion_reason: Option<String>,
    pub failure_fingerprint_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MutationValidityReportV1 {
    pub schema: String,
    pub study_id: String,
    pub analyzer_source_revision: String,
    pub analyzer_worktree_clean: bool,
    pub plan_sha256: String,
    pub receipt_sha256: String,
    pub plan_commitment_sha256: String,
    pub metamorphic_receipt_sha256: String,
    pub planned_variant_count: u64,
    pub observed_variant_count: u64,
    pub completed_evaluation_count: u64,
    pub ineligible_evaluation_count: u64,
    pub planned_adjudication_slots: u64,
    pub observed_adjudication_count: u64,
    pub missing_adjudication_slots: u64,
    pub confirmed_valid_preserving_count: u64,
    pub confirmed_valid_changing_count: u64,
    pub confirmed_invalid_count: u64,
    pub disputed_count: u64,
    pub insufficient_count: u64,
    pub abstained_count: u64,
    pub unreviewed_count: u64,
    pub adjudicable_confirmed_count: u64,
    pub adjudicable_panel_concordance_millionths: Option<f64>,
    pub confirmed_valid_yield_over_planned_millionths: Option<f64>,
    pub pairwise_review_agreement_count: u64,
    pub pairwise_review_disagreement_count: u64,
    pub evaluator_failure_observation_count: u64,
    pub verified_failure_observation_count: u64,
    pub excluded_invalid_mutation_failure_count: u64,
    pub excluded_unconfirmed_mutation_failure_count: u64,
    pub unique_verified_failure_count: u64,
    pub duplicate_verified_failure_observation_count: u64,
    pub total_generator_call_count: u64,
    pub total_generator_token_count: u64,
    pub total_semantic_assistance_call_count: u64,
    pub total_semantic_assistance_token_count: u64,
    pub total_evaluator_call_count: u64,
    pub total_evaluator_token_count: u64,
    pub total_human_review_seconds: u64,
    pub total_wall_time_milliseconds: u64,
    pub total_token_count: u64,
    pub tokens_per_confirmed_valid_mutation: Option<f64>,
    pub tokens_per_unique_verified_failure: Option<f64>,
    pub human_review_seconds_per_unique_verified_failure: Option<f64>,
    pub variant_summaries: Vec<MutationValiditySummaryV1>,
    pub raw_artifact_custody_proven: bool,
    pub reviewer_identity_authenticated: bool,
    pub reviewer_blinding_proven: bool,
    pub panel_reference_validity_proven: bool,
    pub evaluator_execution_proven: bool,
    pub causal_effect_identified: bool,
    pub generalized_validity_estimated: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub claim_boundary: String,
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

fn nonempty(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
}

fn checked_add(total: &mut u64, value: u64, label: &str) -> Result<(), String> {
    *total = total.checked_add(value)
        .ok_or_else(|| format!("{label} counter overflow"))?;
    Ok(())
}

fn plan_commitment(fields: &MutationValidityPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize mutation validity plan: {error}"))?;
    digest_domain(PLAN_DOMAIN, &payload)
}

fn validate_artifacts(
    mutation_id: &str,
    label: &str,
    artifacts: &[MutationArtifactCommitmentV1],
    allow_zero: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    if artifacts.is_empty() || artifacts.len() > MAX_ARTIFACTS {
        issues.push(format!("{mutation_id} {label} must contain 1..={MAX_ARTIFACTS} artifact commitments"));
    }
    let mut ids = BTreeSet::new();
    for artifact in artifacts {
        if !nonempty(&artifact.artifact_id, 128) || !ids.insert(artifact.artifact_id.as_str()) {
            issues.push(format!("{mutation_id} {label} contains an empty or duplicate artifact ID"));
        }
        if !valid_digest(&artifact.content_sha256, allow_zero) {
            issues.push(format!("{mutation_id} {label} contains an invalid artifact content digest"));
        }
        if artifact.byte_length == 0 {
            issues.push(format!("{mutation_id} {label} contains a zero-byte artifact"));
        }
    }
    issues
}

pub fn validate_mutation_validity_plan_fields(
    fields: &MutationValidityPlanFieldsV1,
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();

    if fields.schema != MUTATION_PLAN_SCHEMA {
        issues.push("unsupported mutation validity plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push("analyzer_source_revision must match the exact compiled validator revision".to_owned());
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues.push("qualified mutation validity plans require a clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.study_id, 128) {
        issues.push("study_id must be non-empty and at most 128 bytes".to_owned());
    }
    for (label, value) in [
        ("generator_source_revision", &fields.generator_source_revision),
        ("evaluator_source_revision", &fields.evaluator_source_revision),
    ] {
        if !valid_revision(value) {
            issues.push(format!("{label} must be a full nonzero lowercase Git revision"));
        }
    }
    for (label, value) in [
        ("generator_policy_sha256", &fields.generator_policy_sha256),
        ("semantic_task_set_sha256", &fields.semantic_task_set_sha256),
        ("review_protocol_sha256", &fields.review_protocol_sha256),
        ("metamorphic_plan_sha256", &fields.metamorphic_plan_sha256),
        ("evaluator_prompt_sha256", &fields.evaluator_prompt_sha256),
        ("rubric_sha256", &fields.rubric_sha256),
    ] {
        if !valid_digest(value, allow_zero) {
            issues.push(format!("{label} must be a valid SHA-256 digest"));
        }
    }
    if !(2..=MAX_REVIEWERS as u32).contains(&fields.reviewers_per_mutation) {
        issues.push(format!("reviewers_per_mutation must be in 2..={MAX_REVIEWERS}"));
    }
    if fields.variants.is_empty() || fields.variants.len() > MAX_VARIANTS {
        issues.push(format!("variants must contain 1..={MAX_VARIANTS} entries"));
    }

    let mut mutation_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    for variant in &fields.variants {
        if !nonempty(&variant.mutation_id, 128) || !mutation_ids.insert(variant.mutation_id.as_str()) {
            issues.push("mutation IDs must be non-empty and unique".to_owned());
        }
        if !nonempty(&variant.metamorphic_pair_id, 128) || !pair_ids.insert(variant.metamorphic_pair_id.as_str()) {
            issues.push(format!("{} has an empty or duplicate metamorphic_pair_id", variant.mutation_id));
        }
        if !valid_digest(&variant.baseline_input_sha256, allow_zero)
            || !valid_digest(&variant.variant_input_sha256, allow_zero)
        {
            issues.push(format!("{} has an invalid baseline/variant input digest", variant.mutation_id));
        }
        if variant.baseline_input_sha256 == variant.variant_input_sha256 {
            issues.push(format!("{} baseline and variant input digests must differ", variant.mutation_id));
        }
        if variant.intended_relation == MutationSemanticRelationV1::Changing
            && !variant.target_dimension.as_deref().is_some_and(|value| nonempty(value, 128))
        {
            issues.push(format!("{} meaning-changing mutation requires a target_dimension", variant.mutation_id));
        }
        if variant.intended_relation == MutationSemanticRelationV1::Preserving
            && variant.target_dimension.as_deref().is_some_and(|value| !value.trim().is_empty())
        {
            issues.push(format!("{} semantics-preserving mutation must not declare a changed target_dimension", variant.mutation_id));
        }
        issues.extend(validate_artifacts(&variant.mutation_id, "baseline_artifacts", &variant.baseline_artifacts, allow_zero));
        issues.extend(validate_artifacts(&variant.mutation_id, "variant_artifacts", &variant.variant_artifacts, allow_zero));
    }

    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

pub fn compute_mutation_validity_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: MutationValidityPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("mutation validity plan fields invalid: {error}"))?;
    validate_mutation_validity_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    plan_commitment(&fields)
}

pub fn verify_mutation_validity_plan(
    plan_bytes: &[u8],
) -> Result<(MutationValidityPlanV1, String), String> {
    let plan: MutationValidityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("mutation validity plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_mutation_validity_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    let expected = plan_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("mutation validity plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

fn derive_review_status(
    intended: MutationSemanticRelationV1,
    ratings: &[SemanticAdjudicationV1],
    required: u32,
) -> DerivedReviewStatus {
    if ratings.is_empty() {
        return DerivedReviewStatus::Unreviewed;
    }
    if ratings.len() < required as usize {
        return DerivedReviewStatus::Insufficient;
    }
    let labels: BTreeSet<SemanticReviewDecisionV1> =
        ratings.iter().map(|rating| rating.decision).collect();
    let expected = match intended {
        MutationSemanticRelationV1::Preserving => SemanticReviewDecisionV1::PreservesSemantics,
        MutationSemanticRelationV1::Changing => SemanticReviewDecisionV1::ChangesSemantics,
    };
    let opposite = match intended {
        MutationSemanticRelationV1::Preserving => SemanticReviewDecisionV1::ChangesSemantics,
        MutationSemanticRelationV1::Changing => SemanticReviewDecisionV1::PreservesSemantics,
    };
    if labels.len() == 1 && labels.contains(&expected) {
        return match intended {
            MutationSemanticRelationV1::Preserving => DerivedReviewStatus::ConfirmedValidPreserving,
            MutationSemanticRelationV1::Changing => DerivedReviewStatus::ConfirmedValidChanging,
        };
    }
    if labels.len() == 1 && labels.contains(&opposite) {
        return DerivedReviewStatus::ConfirmedInvalid;
    }
    if labels.contains(&SemanticReviewDecisionV1::PreservesSemantics)
        && labels.contains(&SemanticReviewDecisionV1::ChangesSemantics)
    {
        return DerivedReviewStatus::Disputed;
    }
    if labels == BTreeSet::from([SemanticReviewDecisionV1::Abstain]) {
        return DerivedReviewStatus::Abstained;
    }
    DerivedReviewStatus::Insufficient
}

fn review_status_name(status: DerivedReviewStatus) -> &'static str {
    match status {
        DerivedReviewStatus::ConfirmedValidPreserving => "confirmed_valid_preserving",
        DerivedReviewStatus::ConfirmedValidChanging => "confirmed_valid_changing",
        DerivedReviewStatus::ConfirmedInvalid => "confirmed_invalid",
        DerivedReviewStatus::Disputed => "disputed",
        DerivedReviewStatus::Insufficient => "insufficient",
        DerivedReviewStatus::Abstained => "abstained",
        DerivedReviewStatus::Unreviewed => "unreviewed",
    }
}

fn add_total(
    totals: &mut BTreeMap<&'static str, u64>,
    key: &'static str,
    value: u64,
) -> Result<(), String> {
    let counter = totals.entry(key).or_default();
    checked_add(counter, value, key)
}

pub fn analyze_mutation_validity(
    plan_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<MutationValidityReportV1, String> {
    let (plan, plan_sha256) = verify_mutation_validity_plan(plan_bytes)?;
    let receipt: MutationValidityReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("mutation validity receipt invalid: {error}"))?;
    if receipt.schema != MUTATION_RECEIPT_SCHEMA {
        return Err("unsupported mutation validity receipt schema".to_owned());
    }
    if receipt.analyzer_source_revision != plan.analyzer_source_revision
        || receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        return Err("receipt analyzer revision differs from the exact compiled validator revision".to_owned());
    }
    if receipt.purpose != plan.purpose {
        return Err("receipt purpose differs from the plan".to_owned());
    }
    if !nonempty(&receipt.run_id, 128) {
        return Err("run_id must be non-empty and at most 128 bytes".to_owned());
    }
    if !valid_digest(&receipt.metamorphic_receipt_sha256, !plan.purpose.qualified()) {
        return Err("metamorphic_receipt_sha256 must be a valid digest".to_owned());
    }
    if receipt.plan_sha256 != plan_sha256
        || receipt.plan_commitment_sha256 != plan.plan_commitment_sha256
    {
        return Err("receipt does not bind exact plan bytes and commitment".to_owned());
    }
    if receipt.generator_source_revision != plan.generator_source_revision
        || receipt.evaluator_source_revision != plan.evaluator_source_revision
        || receipt.evaluator_prompt_sha256 != plan.evaluator_prompt_sha256
        || receipt.rubric_sha256 != plan.rubric_sha256
    {
        return Err("receipt generator/evaluator/prompt/rubric differs from frozen plan".to_owned());
    }

    let planned: BTreeMap<&str, &MutationVariantPlanV1> = plan.variants.iter()
        .map(|variant| (variant.mutation_id.as_str(), variant))
        .collect();
    let mut observed = BTreeMap::new();
    for observation in &receipt.observations {
        if observed.insert(observation.mutation_id.as_str(), observation).is_some() {
            return Err(format!("duplicate mutation observation {}", observation.mutation_id));
        }
        let variant = planned.get(observation.mutation_id.as_str())
            .ok_or_else(|| format!("unexpected mutation observation {}", observation.mutation_id))?;
        if observation.baseline_input_sha256 != variant.baseline_input_sha256
            || observation.variant_input_sha256 != variant.variant_input_sha256
        {
            return Err(format!("mutation {} input digest differs from committed plan", observation.mutation_id));
        }
        if observation.adjudications.len() > plan.reviewers_per_mutation as usize {
            return Err(format!("mutation {} has more adjudications than planned reviewer slots", observation.mutation_id));
        }
        let mut reviewer_ids = BTreeSet::new();
        let mut independence_groups = BTreeSet::new();
        for adjudication in &observation.adjudications {
            for (label, digest) in [
                ("adjudicator_id_sha256", &adjudication.adjudicator_id_sha256),
                ("independence_group_sha256", &adjudication.independence_group_sha256),
                ("rationale_sha256", &adjudication.rationale_sha256),
            ] {
                if !valid_digest(digest, false) {
                    return Err(format!("mutation {} has invalid {label}", observation.mutation_id));
                }
            }
            if !reviewer_ids.insert(adjudication.adjudicator_id_sha256.as_str()) {
                return Err(format!("mutation {} reuses an adjudicator identity", observation.mutation_id));
            }
            if !independence_groups.insert(adjudication.independence_group_sha256.as_str()) {
                return Err(format!("mutation {} reuses an adjudicator independence group", observation.mutation_id));
            }
            if adjudication.confidence_millionths > 1_000_000 {
                return Err(format!("mutation {} has out-of-range adjudicator confidence", observation.mutation_id));
            }
        }
        match observation.evaluation_status {
            MutationEvaluationStatusV1::Completed => {
                let passed = observation.evaluator_relation_pass
                    .ok_or_else(|| format!("completed mutation {} requires evaluator_relation_pass", observation.mutation_id))?;
                match (passed, observation.failure_fingerprint_sha256.as_deref()) {
                    (true, None) => {}
                    (false, Some(digest)) if valid_digest(digest, false) => {}
                    (false, _) => return Err(format!("failed mutation {} requires a valid failure fingerprint", observation.mutation_id)),
                    (true, Some(_)) => return Err(format!("passing mutation {} must not carry a failure fingerprint", observation.mutation_id)),
                }
            }
            _ => {
                if observation.evaluator_relation_pass.is_some()
                    || observation.failure_fingerprint_sha256.is_some()
                {
                    return Err(format!("ineligible mutation {} must not carry an evaluator outcome", observation.mutation_id));
                }
            }
        }
    }
    let missing: Vec<&str> = planned.keys().filter(|id| !observed.contains_key(**id)).copied().collect();
    if !missing.is_empty() {
        return Err(format!("receipt is missing {} planned mutation observations", missing.len()));
    }

    let mut totals = BTreeMap::new();
    let mut summaries = Vec::with_capacity(plan.variants.len());
    let mut confirmed_valid_preserving = 0_u64;
    let mut confirmed_valid_changing = 0_u64;
    let mut confirmed_invalid = 0_u64;
    let mut disputed = 0_u64;
    let mut insufficient = 0_u64;
    let mut abstained = 0_u64;
    let mut unreviewed = 0_u64;
    let mut completed = 0_u64;
    let mut ineligible = 0_u64;
    let mut observed_adjudications = 0_u64;
    let mut pairwise_agreements = 0_u64;
    let mut pairwise_disagreements = 0_u64;
    let mut failure_observations = 0_u64;
    let mut verified_failure_observations = 0_u64;
    let mut excluded_invalid_failures = 0_u64;
    let mut excluded_unconfirmed_failures = 0_u64;
    let mut verified_fingerprints = BTreeSet::new();
    let mut duplicate_verified_failures = 0_u64;

    for variant in &plan.variants {
        let observation = observed.get(variant.mutation_id.as_str()).unwrap();
        let status = derive_review_status(
            variant.intended_relation,
            &observation.adjudications,
            plan.reviewers_per_mutation,
        );
        checked_add(&mut observed_adjudications, observation.adjudications.len() as u64, "observed adjudication")?;
        let pair_count = (observation.adjudications.len() as u64)
            .checked_mul(observation.adjudications.len().saturating_sub(1) as u64)
            .map(|value| value / 2)
            .ok_or_else(|| "reviewer pair count overflow".to_owned())?;
        let mut agreements = 0_u64;
        let mut disagreements = 0_u64;
        for left in 0..observation.adjudications.len() {
            for right in (left + 1)..observation.adjudications.len() {
                if observation.adjudications[left].decision == observation.adjudications[right].decision {
                    agreements += 1;
                } else {
                    disagreements += 1;
                }
            }
        }
        debug_assert_eq!(agreements + disagreements, pair_count);
        checked_add(&mut pairwise_agreements, agreements, "pairwise review agreement")?;
        checked_add(&mut pairwise_disagreements, disagreements, "pairwise review disagreement")?;

        match status {
            DerivedReviewStatus::ConfirmedValidPreserving => confirmed_valid_preserving += 1,
            DerivedReviewStatus::ConfirmedValidChanging => confirmed_valid_changing += 1,
            DerivedReviewStatus::ConfirmedInvalid => confirmed_invalid += 1,
            DerivedReviewStatus::Disputed => disputed += 1,
            DerivedReviewStatus::Insufficient => insufficient += 1,
            DerivedReviewStatus::Abstained => abstained += 1,
            DerivedReviewStatus::Unreviewed => unreviewed += 1,
        }
        let valid_mutation = matches!(
            status,
            DerivedReviewStatus::ConfirmedValidPreserving | DerivedReviewStatus::ConfirmedValidChanging
        );
        let is_invalid = status == DerivedReviewStatus::ConfirmedInvalid;
        let mut failure_counted = false;
        let mut exclusion_reason = None;
        if observation.evaluation_status == MutationEvaluationStatusV1::Completed {
            completed += 1;
            if observation.evaluator_relation_pass == Some(false) {
                failure_observations += 1;
                let fingerprint = observation.failure_fingerprint_sha256.as_ref().unwrap();
                if valid_mutation {
                    verified_failure_observations += 1;
                    failure_counted = true;
                    if !verified_fingerprints.insert(fingerprint.clone()) {
                        duplicate_verified_failures += 1;
                    }
                } else if is_invalid {
                    excluded_invalid_failures += 1;
                    exclusion_reason = Some("mutation_confirmed_invalid".to_owned());
                } else {
                    excluded_unconfirmed_failures += 1;
                    exclusion_reason = Some(review_status_name(status).to_owned());
                }
            }
        } else {
            ineligible += 1;
        }

        let summary = MutationValiditySummaryV1 {
            mutation_id: variant.mutation_id.clone(),
            intended_relation: variant.intended_relation,
            review_count: observation.adjudications.len() as u64,
            pairwise_agreement_count: agreements,
            pairwise_disagreement_count: disagreements,
            review_status: review_status_name(status).to_owned(),
            evaluation_status: observation.evaluation_status,
            evaluator_relation_pass: observation.evaluator_relation_pass,
            failure_counted_as_verified: failure_counted,
            failure_exclusion_reason: exclusion_reason,
            failure_fingerprint_sha256: if failure_counted { observation.failure_fingerprint_sha256.clone() } else { None },
        };
        summaries.push(summary);

        for (key, value) in [
            ("generator_calls", observation.generator_call_count),
            ("generator_tokens", observation.generator_token_count),
            ("semantic_assistance_calls", observation.semantic_assistance_call_count),
            ("semantic_assistance_tokens", observation.semantic_assistance_token_count),
            ("evaluator_calls", observation.evaluator_call_count),
            ("evaluator_tokens", observation.evaluator_token_count),
            ("human_review_seconds", observation.human_review_seconds),
            ("wall_time_milliseconds", observation.wall_time_milliseconds),
        ] {
            add_total(&mut totals, key, value)?;
        }
    }

    let planned_count = plan.variants.len() as u64;
    let planned_slots = planned_count.checked_mul(u64::from(plan.reviewers_per_mutation))
        .ok_or_else(|| "planned adjudication slot count overflow".to_owned())?;
    let missing_slots = planned_slots.saturating_sub(observed_adjudications);
    let adjudicable = confirmed_valid_preserving + confirmed_valid_changing + confirmed_invalid;
    let confirmed_valid = confirmed_valid_preserving + confirmed_valid_changing;
    let adjudicable_concordance = if adjudicable == 0 {
        None
    } else {
        Some(confirmed_valid as f64 / adjudicable as f64 * 1_000_000.0)
    };
    let valid_yield = Some(confirmed_valid as f64 / planned_count as f64 * 1_000_000.0);
    let total_tokens = totals.get("generator_tokens").copied().unwrap_or(0)
        .checked_add(totals.get("semantic_assistance_tokens").copied().unwrap_or(0))
        .and_then(|value| value.checked_add(totals.get("evaluator_tokens").copied().unwrap_or(0)))
        .ok_or_else(|| "total token count overflow".to_owned())?;
    let unique_verified = verified_fingerprints.len() as u64;

    let fields = MutationValidityReportV1 {
        schema: MUTATION_REPORT_SCHEMA.to_owned(),
        study_id: plan.study_id,
        analyzer_source_revision: plan.analyzer_source_revision,
        analyzer_worktree_clean: crate::BUILT_GENERATOR_WORKTREE_CLEAN == "true",
        plan_sha256,
        receipt_sha256: sha256_hex(receipt_bytes),
        plan_commitment_sha256: plan.plan_commitment_sha256,
        metamorphic_receipt_sha256: receipt.metamorphic_receipt_sha256,
        planned_variant_count: planned_count,
        observed_variant_count: receipt.observations.len() as u64,
        completed_evaluation_count: completed,
        ineligible_evaluation_count: ineligible,
        planned_adjudication_slots: planned_slots,
        observed_adjudication_count: observed_adjudications,
        missing_adjudication_slots: missing_slots,
        confirmed_valid_preserving_count: confirmed_valid_preserving,
        confirmed_valid_changing_count: confirmed_valid_changing,
        confirmed_invalid_count: confirmed_invalid,
        disputed_count: disputed,
        insufficient_count: insufficient,
        abstained_count: abstained,
        unreviewed_count: unreviewed,
        adjudicable_confirmed_count: adjudicable,
        adjudicable_panel_concordance_millionths: adjudicable_concordance,
        confirmed_valid_yield_over_planned_millionths: valid_yield,
        pairwise_review_agreement_count: pairwise_agreements,
        pairwise_review_disagreement_count: pairwise_disagreements,
        evaluator_failure_observation_count: failure_observations,
        verified_failure_observation_count: verified_failure_observations,
        excluded_invalid_mutation_failure_count: excluded_invalid_failures,
        excluded_unconfirmed_mutation_failure_count: excluded_unconfirmed_failures,
        unique_verified_failure_count: unique_verified,
        duplicate_verified_failure_observation_count: duplicate_verified_failures,
        total_generator_call_count: totals.get("generator_calls").copied().unwrap_or(0),
        total_generator_token_count: totals.get("generator_tokens").copied().unwrap_or(0),
        total_semantic_assistance_call_count: totals.get("semantic_assistance_calls").copied().unwrap_or(0),
        total_semantic_assistance_token_count: totals.get("semantic_assistance_tokens").copied().unwrap_or(0),
        total_evaluator_call_count: totals.get("evaluator_calls").copied().unwrap_or(0),
        total_evaluator_token_count: totals.get("evaluator_tokens").copied().unwrap_or(0),
        total_human_review_seconds: totals.get("human_review_seconds").copied().unwrap_or(0),
        total_wall_time_milliseconds: totals.get("wall_time_milliseconds").copied().unwrap_or(0),
        total_token_count: total_tokens,
        tokens_per_confirmed_valid_mutation: if confirmed_valid == 0 { None } else { Some(total_tokens as f64 / confirmed_valid as f64) },
        tokens_per_unique_verified_failure: if unique_verified == 0 { None } else { Some(total_tokens as f64 / unique_verified as f64) },
        human_review_seconds_per_unique_verified_failure: if unique_verified == 0 { None } else { Some(totals.get("human_review_seconds").copied().unwrap_or(0) as f64 / unique_verified as f64) },
        variant_summaries: summaries,
        raw_artifact_custody_proven: false,
        reviewer_identity_authenticated: false,
        reviewer_blinding_proven: false,
        panel_reference_validity_proven: false,
        evaluator_execution_proven: false,
        causal_effect_identified: false,
        generalized_validity_estimated: false,
        external_anchor_required: true,
        scoring_eligible_declared: false,
        claim_boundary: "The report validates declared artifact commitments, independent reviewer commitments, adjudication outcomes, evaluator relation outcomes and resource counters for a supplied fixed receipt. It does not custody raw artifact bytes, authenticate reviewers, prove blinding, establish that the panel is a valid reference, authenticate evaluator execution, identify causal effects, estimate generalized validity, or grant scoring eligibility.".to_owned(),
    };
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| format!("cannot serialize mutation validity report: {error}"))?;
    let _report_commitment = digest_domain(REPORT_DOMAIN, &payload)?;
    Ok(fields)
}
